import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import moderngl
import glfw
import numpy as np
import time
import threading
from queue import Queue
import json
import numpy as np
import time
import threading
import psutil
import GPUtil


class PointCloudStreamRenderer:
    def __init__(self, width=1920, height=1080, max_points=5000000):
        self.width = width
        self.height = height
        self.max_points = max_points
        self.current_point_count = 0

        # Streaming properties
        self.stream_queue = Queue(maxsize=100)
        self.is_streaming = False
        self.stream_thread = None

        # Camera control
        self.camera_distance = 10.0
        self.camera_rotation = [0.0, 0.0]
        self.mouse_dragging = False
        self.last_mouse_pos = [0, 0]

        # Animation
        self.start_time = time.time()
        self.frame_count = 0

        # CPU buffers for data transfer
        self.cpu_positions = np.zeros((max_points, 3), dtype=np.float32)
        self.cpu_colors = np.zeros((max_points, 3), dtype=np.float32)

        # Initialize systems
        self.setup_glfw()
        self.setup_moderngl()
        self.setup_cuda_kernels()
        self.setup_buffers()
        self.setup_shaders()

        # Start with empty point cloud
        self.initialize_empty_point_cloud()

    def setup_glfw(self):
        """Initialize GLFW window with input callbacks"""
        if not glfw.init():
            raise Exception("GLFW initialization failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.SAMPLES, 8)  # 8x MSAA

        self.window = glfw.create_window(
            self.width, self.height,
            f"Real-time Point Cloud Stream - GPU Accelerated",
            None, None
        )

        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        glfw.make_context_current(self.window)

        # Set up input callbacks
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_key_callback(self.window, self.key_callback)

    def setup_moderngl(self):
        """Initialize ModernGL context"""
        self.ctx = moderngl.create_context()

        # Enable rendering features
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def setup_cuda_kernels(self):
        """Compile CUDA kernels for point cloud processing"""
        cuda_code = """
        #include <cuda_runtime.h>
        #include <math.h>

        // Kernel to update existing points with new data
        __global__ void update_points_kernel(
            float3* positions, 
            float3* colors,
            float3* new_positions,
            float3* new_colors,
            int* point_count,
            int new_points_count,
            int max_points,
            float time
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < new_points_count) {
                int target_idx = (*point_count + idx) % max_points;

                // Copy new point data
                positions[target_idx] = new_positions[idx];

                // Animate colors based on time
                colors[target_idx].x = new_colors[idx].x * (0.8f + 0.2f * sinf(time + idx * 0.1f));
                colors[target_idx].y = new_colors[idx].y * (0.8f + 0.2f * cosf(time + idx * 0.1f));
                colors[target_idx].z = new_colors[idx].z * (0.8f + 0.2f * sinf(time * 2.0f + idx * 0.1f));
            }

            // Update point count if this is the last thread
            if (idx == 0 && new_points_count > 0) {
                *point_count = min(max_points, *point_count + new_points_count);
            }
        }

        // Kernel for dynamic point animation
        __global__ void animate_points_kernel(
            float3* positions,
            float3* colors,
            int point_count,
            float time,
            float delta_time
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < point_count) {
                // Gentle floating animation
                float3 pos = positions[idx];

                // Per-point oscillation
                float freq = 0.5f + fmodf(idx * 0.001f, 2.0f);
                float amp = 0.02f;

                pos.y += sinf(time * freq + idx) * amp * delta_time;
                pos.x += cosf(time * freq * 0.7f + idx) * amp * delta_time;

                positions[idx] = pos;

                // Pulsing color effect
                float pulse = (sinf(time * 2.0f + idx * 0.01f) + 1.0f) * 0.1f;
                colors[idx].x = fminf(1.0f, colors[idx].x + pulse);
                colors[idx].y = fminf(1.0f, colors[idx].y + pulse * 0.7f);
            }
        }

        // Kernel to remove old points (FIFO behavior)
        __global__ void remove_old_points_kernel(
            float3* positions,
            float3* colors,
            int* point_count,
            int points_to_remove,
            int max_points
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < points_to_remove && idx < *point_count) {
                // Shift points down (FIFO removal)
                if (idx + points_to_remove < *point_count) {
                    positions[idx] = positions[idx + points_to_remove];
                    colors[idx] = colors[idx + points_to_remove];
                } else {
                    // Clear remaining points
                    positions[idx] = make_float3(0.0f, 0.0f, 0.0f);
                    colors[idx] = make_float3(0.0f, 0.0f, 0.0f);
                }
            }

            // Update point count
            if (idx == 0 && points_to_remove > 0) {
                *point_count = max(0, *point_count - points_to_remove);
            }
        }

        // Kernel to copy data to CPU buffer for OpenGL
        __global__ void copy_to_cpu_kernel(
            float3* device_positions,
            float3* device_colors,
            float* cpu_positions,  // Flattened array
            float* cpu_colors,     // Flattened array
            int point_count
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < point_count) {
                // Copy positions (convert float3 to flattened float array)
                cpu_positions[idx * 3 + 0] = device_positions[idx].x;
                cpu_positions[idx * 3 + 1] = device_positions[idx].y;
                cpu_positions[idx * 3 + 2] = device_positions[idx].z;

                // Copy colors (convert float3 to flattened float array)
                cpu_colors[idx * 3 + 0] = device_colors[idx].x;
                cpu_colors[idx * 3 + 1] = device_colors[idx].y;
                cpu_colors[idx * 3 + 2] = device_colors[idx].z;
            }
        }
        """

        self.mod = SourceModule(cuda_code, options=['-use_fast_math'])
        self.update_kernel = self.mod.get_function("update_points_kernel")
        self.animate_kernel = self.mod.get_function("animate_points_kernel")
        self.remove_kernel = self.mod.get_function("remove_old_points_kernel")
        self.copy_to_cpu_kernel = self.mod.get_function("copy_to_cpu_kernel")

    def setup_buffers(self):
        """Setup CUDA and OpenGL buffers"""
        # CUDA device memory
        self.d_positions = cuda.mem_alloc(self.max_points * 12)  # float3 = 12 bytes
        self.d_colors = cuda.mem_alloc(self.max_points * 12)
        self.d_point_count = cuda.mem_alloc(4)  # int

        # CPU buffers for OpenGL data transfer
        self.d_cpu_positions = cuda.mem_alloc(self.max_points * 12)  # For GPU->CPU transfer
        self.d_cpu_colors = cuda.mem_alloc(self.max_points * 12)  # For GPU->CPU transfer

        # Initialize with zeros
        cuda.memset_d32(self.d_positions, 0, self.max_points * 3)
        cuda.memset_d32(self.d_colors, 0, self.max_points * 3)
        cuda.memset_d32(self.d_cpu_positions, 0, self.max_points * 3)
        cuda.memset_d32(self.d_cpu_colors, 0, self.max_points * 3)

        # Initialize point count
        zero_count = np.array([0], dtype=np.int32)
        cuda.memcpy_htod(self.d_point_count, zero_count)

        # OpenGL buffers - using dynamic draw since we update frequently
        self.vbo_positions = self.ctx.buffer(reserve=self.max_points * 12)
        self.vbo_colors = self.ctx.buffer(reserve=self.max_points * 12)

    def setup_shaders(self):
        """Setup ModernGL shaders for point cloud rendering"""
        vertex_shader = """
            #version 430 core

            layout (location = 0) in vec3 in_position;
            layout (location = 1) in vec3 in_color;

            out vec3 vs_color;
            out float vs_depth;
            out float vs_point_size;

            uniform mat4 mvp;
            uniform mat4 view;
            uniform float time;
            uniform int total_points;
            uniform vec3 camera_pos;
            uniform float point_scale;

            void main() {
                vec4 world_pos = vec4(in_position, 1.0);
                vec4 view_pos = view * world_pos;
                gl_Position = mvp * world_pos;

                // Depth-based point size with scaling
                vs_depth = length(view_pos.xyz);
                float base_size = point_scale * 5.0;
                float distance_scale = max(0.5, 15.0 / (vs_depth * 0.3));
                vs_point_size = base_size * distance_scale;

                // Size based on point density
                float density_scale = 1.0 - (float(gl_VertexID) / float(total_points)) * 0.3;
                gl_PointSize = vs_point_size * density_scale;

                // Color with time-based pulsation
                float pulse = (sin(time * 3.0 + float(gl_VertexID) * 0.001) + 1.0) * 0.1;
                vs_color = in_color * (0.9 + pulse);
            }
        """

        fragment_shader = """
            #version 430 core

            in vec3 vs_color;
            in float vs_depth;
            in float vs_point_size;
            out vec4 frag_color;

            uniform float time;
            uniform int render_mode;

            void main() {
                // Circular point with smooth edge
                vec2 coord = gl_PointCoord * 2.0 - 1.0;
                float dist = length(coord);

                if (dist > 1.0) {
                    discard;
                }

                // Smooth alpha falloff
                float alpha = 1.0 - smoothstep(0.8, 1.0, dist);

                // Depth-based transparency
                alpha *= max(0.3, 1.0 - vs_depth * 0.05);

                // Pulsing brightness
                float brightness = 0.8 + 0.2 * sin(time * 2.0);

                if (render_mode == 1) {
                    // Heatmap mode
                    float intensity = (vs_color.r + vs_color.g + vs_color.b) / 3.0;
                    vec3 heat_color = vec3(
                        intensity * 2.0,
                        intensity * 1.5,
                        (1.0 - intensity) * 2.0
                    );
                    frag_color = vec4(heat_color * brightness, alpha);
                } else {
                    // Normal color mode
                    frag_color = vec4(vs_color * brightness, alpha);
                }
            }
        """

        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )

        # Create vertex array
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (self.vbo_positions, '3f', 'in_position'),
                (self.vbo_colors, '3f', 'in_color')
            ]
        )

    def initialize_empty_point_cloud(self):
        """Initialize with an empty point cloud"""
        # Update OpenGL buffers with initial empty data
        empty_positions = np.zeros((self.max_points, 3), dtype=np.float32)
        empty_colors = np.zeros((self.max_points, 3), dtype=np.float32)

        self.vbo_positions.write(empty_positions.tobytes())
        self.vbo_colors.write(empty_colors.tobytes())

    def add_point_cloud_chunk(self, positions, colors):
        """Add a chunk of points to the stream"""
        if not self.is_streaming:
            return

        if positions.shape[0] != colors.shape[0]:
            raise ValueError("Positions and colors must have same length")

        chunk = {
            'positions': positions.astype(np.float32),
            'colors': colors.astype(np.float32),
            'timestamp': time.time()
        }

        try:
            self.stream_queue.put_nowait(chunk)
        except:
            # Queue full, remove oldest if needed
            try:
                self.stream_queue.get_nowait()
                self.stream_queue.put_nowait(chunk)
            except:
                pass

    def process_stream_queue(self):
        """Process incoming point cloud chunks from the queue"""
        processed_chunks = 0
        total_points = 0

        while not self.stream_queue.empty() and processed_chunks < 10:  # Limit chunks per frame
            try:
                chunk = self.stream_queue.get_nowait()
                self.process_point_chunk(chunk)
                processed_chunks += 1
                total_points += chunk['positions'].shape[0]
            except:
                break

        return total_points

    def process_point_chunk(self, chunk):
        """Process a single point cloud chunk using CUDA"""
        positions = chunk['positions']
        colors = chunk['colors']
        num_points = positions.shape[0]

        if num_points == 0:
            return

        # Allocate temporary device memory for new points
        d_new_positions = cuda.mem_alloc(num_points * 12)
        d_new_colors = cuda.mem_alloc(num_points * 12)

        # Copy new data to device
        cuda.memcpy_htod(d_new_positions, positions)
        cuda.memcpy_htod(d_new_colors, colors)

        # Launch update kernel
        block_size = 256
        grid_size = (num_points + block_size - 1) // block_size
        current_time = time.time() - self.start_time

        self.update_kernel(
            self.d_positions, self.d_colors,
            d_new_positions, d_new_colors,
            self.d_point_count, np.int32(num_points),
            np.int32(self.max_points), np.float32(current_time),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        # Free temporary memory
        d_new_positions.free()
        d_new_colors.free()

    def animate_points(self, delta_time):
        """Animate existing points"""
        if self.current_point_count == 0:
            return

        block_size = 256
        grid_size = (self.current_point_count + block_size - 1) // block_size
        current_time = time.time() - self.start_time

        self.animate_kernel(
            self.d_positions, self.d_colors,
            np.int32(self.current_point_count), np.float32(current_time),
            np.float32(delta_time),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

    def remove_old_points(self, points_to_remove):
        """Remove oldest points (FIFO)"""
        if points_to_remove <= 0 or self.current_point_count == 0:
            return

        block_size = 256
        grid_size = (points_to_remove + block_size - 1) // block_size

        self.remove_kernel(
            self.d_positions, self.d_colors,
            self.d_point_count, np.int32(points_to_remove),
            np.int32(self.max_points),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

    def sync_buffers(self):
        """Sync CUDA data with OpenGL buffers using CPU intermediary"""
        if self.current_point_count == 0:
            return

        # Copy data from GPU to CPU buffers using CUDA kernel
        block_size = 256
        grid_size = (self.current_point_count + block_size - 1) // block_size

        self.copy_to_cpu_kernel(
            self.d_positions, self.d_colors,
            self.d_cpu_positions, self.d_cpu_colors,  # These are float* (flattened)
            np.int32(self.current_point_count),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        # Copy from GPU to CPU memory
        cuda.memcpy_dtoh(self.cpu_positions, self.d_cpu_positions)
        cuda.memcpy_dtoh(self.cpu_colors, self.d_cpu_colors)

        # Update OpenGL buffers
        self.vbo_positions.write(self.cpu_positions[:self.current_point_count].tobytes())
        self.vbo_colors.write(self.cpu_colors[:self.current_point_count].tobytes())

    def get_view_matrix(self):
        """Calculate view matrix based on camera controls"""
        # Calculate camera position based on spherical coordinates
        cam_x = self.camera_distance * np.cos(self.camera_rotation[0]) * np.cos(self.camera_rotation[1])
        cam_y = self.camera_distance * np.sin(self.camera_rotation[1])
        cam_z = self.camera_distance * np.sin(self.camera_rotation[0]) * np.cos(self.camera_rotation[1])

        eye = np.array([cam_x, cam_y, cam_z])
        target = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])

        return self.look_at_matrix(eye, target, up)

    def look_at_matrix(self, eye, target, up):
        """Create look-at view matrix"""
        forward = (target - eye)
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        new_up = np.cross(right, forward)

        return np.array([
            [right[0], new_up[0], -forward[0], 0],
            [right[1], new_up[1], -forward[1], 0],
            [right[2], new_up[2], -forward[2], 0],
            [-np.dot(right, eye), -np.dot(new_up, eye), np.dot(forward, eye), 1]
        ], dtype=np.float32)

    def perspective_matrix(self, fov, aspect, near, far):
        """Create perspective projection matrix"""
        f = 1.0 / np.tan(np.radians(fov) / 2)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)

    def render_frame(self, delta_time):
        """Render a single frame"""
        # Process incoming stream data
        new_points = self.process_stream_queue()

        # Update point count from GPU
        count_array = np.zeros(1, dtype=np.int32)
        cuda.memcpy_dtoh(count_array, self.d_point_count)
        self.current_point_count = count_array[0]

        # Animate existing points
        self.animate_points(delta_time)

        # Sync CUDA and OpenGL buffers
        self.sync_buffers()

        # Clear screen
        self.ctx.clear(0.05, 0.05, 0.08)  # Dark blue-gray background

        if self.current_point_count > 0:
            # Calculate matrices
            view = self.get_view_matrix()
            projection = self.perspective_matrix(60, self.width / self.height, 0.1, 1000.0)
            mvp = projection @ view

            # Set shader uniforms
            current_time = time.time() - self.start_time
            camera_pos = np.array([
                self.camera_distance * np.cos(self.camera_rotation[0]) * np.cos(self.camera_rotation[1]),
                self.camera_distance * np.sin(self.camera_rotation[1]),
                self.camera_distance * np.sin(self.camera_rotation[0]) * np.cos(self.camera_rotation[1])
            ], dtype=np.float32)

            self.program['mvp'].write(mvp.astype('f4').tobytes())
            self.program['view'].write(view.astype('f4').tobytes())
            self.program['time'].value = current_time
            self.program['total_points'].value = self.current_point_count
            self.program['camera_pos'].value = tuple(camera_pos)
            self.program['point_scale'].value = 1.0
            self.program['render_mode'].value = 0  # Normal mode

            # Render points
            self.vao.render(moderngl.POINTS, vertices=self.current_point_count)

        return new_points

    def start_streaming(self):
        """Start the point cloud streaming"""
        self.is_streaming = True
        print("Point cloud streaming started...")

    def stop_streaming(self):
        """Stop the point cloud streaming"""
        self.is_streaming = False
        print("Point cloud streaming stopped.")

    def run(self):
        """Main rendering loop"""
        self.start_streaming()

        last_time = time.time()
        fps_history = []

        while not glfw.window_should_close(self.window):
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time

            # Update window size
            new_width, new_height = glfw.get_window_size(self.window)
            if new_width != self.width or new_height != self.height:
                self.width, self.height = new_width, new_height
                self.ctx.viewport = (0, 0, self.width, self.height)

            # Render frame
            new_points = self.render_frame(delta_time)

            # Swap buffers
            glfw.swap_buffers(self.window)
            glfw.poll_events()

            # Update FPS calculation
            self.frame_count += 1
            if self.frame_count % 60 == 0:
                fps = 1.0 / delta_time if delta_time > 0 else 0
                fps_history.append(fps)
                if len(fps_history) > 60:
                    fps_history.pop(0)

                avg_fps = sum(fps_history) / len(fps_history)

                # Update window title with stats
                title = (f"Point Cloud Stream - Points: {self.current_point_count:,} | "
                         f"FPS: {avg_fps:.1f} | New: {new_points}")
                glfw.set_window_title(self.window, title)

        self.stop_streaming()
        glfw.terminate()

    # Input callbacks
    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_dragging = (action == glfw.PRESS)

    def mouse_move_callback(self, window, x, y):
        if self.mouse_dragging:
            dx = (x - self.last_mouse_pos[0]) * 0.01
            dy = (y - self.last_mouse_pos[1]) * 0.01

            self.camera_rotation[0] += dx
            self.camera_rotation[1] = np.clip(self.camera_rotation[1] + dy, -1.5, 1.5)

        self.last_mouse_pos = [x, y]

    def scroll_callback(self, window, x_offset, y_offset):
        self.camera_distance = np.clip(self.camera_distance - y_offset * 0.5, 2.0, 50.0)

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_R:
                # Reset camera
                self.camera_distance = 10.0
                self.camera_rotation = [0.0, 0.0]
            elif key == glfw.KEY_C:
                # Clear point cloud
                self.initialize_empty_point_cloud()
                # Reset point count on GPU
                zero_count = np.array([0], dtype=np.int32)
                cuda.memcpy_htod(self.d_point_count, zero_count)
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

class PointCloudStreamGenerator:
    def __init__(self, renderer, points_per_chunk=10000, stream_rate=30):
        self.renderer = renderer
        self.points_per_chunk = points_per_chunk
        self.stream_rate = stream_rate
        self.generation_thread = None
        self.is_generating = False

        # Generation parameters
        self.current_shape = "sphere"
        self.generation_time = 0.0

    def start_generation(self):
        """Start generating point cloud streams"""
        self.is_generating = True
        self.generation_thread = threading.Thread(target=self.generation_loop)
        self.generation_thread.daemon = True
        self.generation_thread.start()
        print(f"Started point cloud generation: {self.points_per_chunk} points/chunk at {self.stream_rate} Hz")

    def stop_generation(self):
        """Stop generating point cloud streams"""
        self.is_generating = False
        if self.generation_thread:
            self.generation_thread.join()
        print("Point cloud generation stopped.")

    def generation_loop(self):
        """Main generation loop"""
        frame_time = 1.0 / self.stream_rate

        while self.is_generating:
            start_time = time.time()

            # Generate point cloud chunk
            positions, colors = self.generate_point_chunk()

            # Send to renderer
            self.renderer.add_point_cloud_chunk(positions, colors)

            # Maintain stream rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)

            self.generation_time = time.time() - start_time

    def generate_point_chunk(self):
        """Generate a chunk of points with various patterns"""
        if self.current_shape == "sphere":
            return self.generate_sphere_chunk()
        elif self.current_shape == "cube":
            return self.generate_cube_chunk()
        elif self.current_shape == "plane":
            return self.generate_plane_chunk()
        elif self.current_shape == "noise":
            return self.generate_noise_chunk()
        elif self.current_shape == "wave":
            return self.generate_wave_chunk()
        else:
            return self.generate_sphere_chunk()

    def generate_sphere_chunk(self):
        """Generate points on a sphere surface"""
        # Random points on sphere
        phi = np.random.uniform(0, 2 * np.pi, self.points_per_chunk)
        theta = np.arccos(2 * np.random.uniform(0, 1, self.points_per_chunk) - 1)

        radius = 2.0 + np.sin(time.time()) * 0.5  # Animated radius

        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)

        positions = np.column_stack([x, y, z])

        # Colors based on position
        colors = np.column_stack([
            (x + radius) / (2 * radius),
            (y + radius) / (2 * radius),
            (z + radius) / (2 * radius)
        ])

        return positions.astype(np.float32), colors.astype(np.float32)

    def generate_cube_chunk(self):
        """Generate points in a cube"""
        positions = np.random.uniform(-3, 3, (self.points_per_chunk, 3))

        # Colors based on distance from center
        distances = np.linalg.norm(positions, axis=1)
        normalized_dist = distances / np.max(distances)

        colors = np.column_stack([
            normalized_dist,
            1 - normalized_dist,
            np.sin(normalized_dist * np.pi)
        ])

        return positions.astype(np.float32), colors.astype(np.float32)

    def generate_plane_chunk(self):
        """Generate points on a plane with wave animation"""
        x = np.random.uniform(-4, 4, self.points_per_chunk)
        z = np.random.uniform(-4, 4, self.points_per_chunk)

        # Animated wave surface
        wave_time = time.time()
        y = np.sin(x * 2 + wave_time) * np.cos(z * 2 + wave_time) * 0.5

        positions = np.column_stack([x, y, z])

        # Colors based on height
        normalized_y = (y + 1) / 2
        colors = np.column_stack([
            normalized_y,
            np.cos(normalized_y * np.pi),
            np.sin(normalized_y * np.pi * 2)
        ])

        return positions.astype(np.float32), colors.astype(np.float32)

    def generate_noise_chunk(self):
        """Generate 3D noise field points"""
        positions = np.random.uniform(-3, 3, (self.points_per_chunk, 3))

        # Add Perlin-like noise structure
        t = time.time()
        for i in range(3):
            freq = 2.0 + i
            positions[:, i] += np.sin(positions[:, (i + 1) % 3] * freq + t) * 0.3

        # Colors based on noise
        noise_val = np.sin(positions[:, 0] * 3 + t) * np.cos(positions[:, 1] * 3)
        colors = np.column_stack([
            (noise_val + 1) * 0.5,
            (np.cos(noise_val * np.pi) + 1) * 0.5,
            (np.sin(noise_val * 2 * np.pi) + 1) * 0.5
        ])

        return positions.astype(np.float32), colors.astype(np.float32)

    def generate_wave_chunk(self):
        """Generate points in a wave pattern"""
        x = np.random.uniform(-4, 4, self.points_per_chunk)
        z = np.random.uniform(-4, 4, self.points_per_chunk)

        # Complex wave surface
        t = time.time()
        y = (np.sin(x * 3 + t) + np.cos(z * 2 + t * 1.5)) * 0.3

        positions = np.column_stack([x, y, z])

        # Gradient colors
        colors = np.column_stack([
            (x + 4) / 8,
            (y + 1) / 2,
            (z + 4) / 8
        ])

        return positions.astype(np.float32), colors.astype(np.float32)

    def set_shape(self, shape_name):
        """Change the generated shape"""
        if shape_name in ["sphere", "cube", "plane", "noise", "wave"]:
            self.current_shape = shape_name
            print(f"Switched to {shape_name} pattern")


def main():
    # Create renderer with 2 million point capacity
    renderer = PointCloudStreamRenderer(max_points=2000000)

    # Create stream generator
    generator = PointCloudStreamGenerator(
        renderer,
        points_per_chunk=5000,  # 5K points per chunk
        stream_rate=60  # 60 Hz stream rate
    )

    # Start generation in background thread
    generator.start_generation()

    # Cycle through shapes every 10 seconds
    def shape_cycler():
        shapes = ["sphere", "cube", "plane", "noise", "wave"]
        current_idx = 0

        while True:
            time.sleep(10)
            current_idx = (current_idx + 1) % len(shapes)
            generator.set_shape(shapes[current_idx])

    cycle_thread = threading.Thread(target=shape_cycler)
    cycle_thread.daemon = True
    cycle_thread.start()

    try:
        # Start rendering (this blocks)
        renderer.run()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        generator.stop_generation()


class PerformanceMonitor:
    def __init__(self):
        self.gpu_stats = []
        self.cpu_stats = []
        self.memory_stats = []

    def update_stats(self):
        """Update performance statistics"""
        # GPU stats
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            self.gpu_stats.append({
                'load': gpu.load,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'temperature': gpu.temperature
            })

        # CPU stats
        self.cpu_stats.append(psutil.cpu_percent())

        # Memory stats
        memory = psutil.virtual_memory()
        self.memory_stats.append(memory.percent)

        # Keep only recent history
        max_history = 100
        for stats in [self.gpu_stats, self.cpu_stats, self.memory_stats]:
            if len(stats) > max_history:
                del stats[:len(stats) - max_history]

    def print_summary(self):
        """Print performance summary"""
        if not self.gpu_stats or not self.cpu_stats:
            return

        latest_gpu = self.gpu_stats[-1]
        avg_cpu = np.mean(self.cpu_stats[-10:])
        avg_memory = np.mean(self.memory_stats[-10:])

        print(f"GPU Load: {latest_gpu['load'] * 100:.1f}% | "
              f"GPU Memory: {latest_gpu['memory_used']}/{latest_gpu['memory_total']}MB | "
              f"CPU: {avg_cpu:.1f}% | RAM: {avg_memory:.1f}%")


if __name__ == "__main__":
    main()