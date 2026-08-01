import cv2
import numpy as np
import pandas as pd
import gradio as gr
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import time
import os

# Load color CSV
try:
    df = pd.read_csv("colors.csv")
    print("✅ Colors CSV loaded successfully")
except FileNotFoundError:
    print("❌ colors.csv not found. Creating basic color dataset.")
    # Create a more comprehensive basic color dataset
    df = pd.DataFrame({
        'color_name': ['Red', 'Green', 'Blue', 'White', 'Black', 'Yellow', 'Cyan', 'Magenta', 
                      'Orange', 'Purple', 'Pink', 'Brown', 'Gray', 'Navy', 'Lime', 'Maroon'],
        'R': [255, 0, 0, 255, 0, 255, 0, 255, 255, 128, 255, 165, 128, 0, 0, 128],
        'G': [0, 255, 0, 255, 0, 255, 255, 0, 165, 0, 192, 42, 128, 0, 255, 0],
        'B': [0, 0, 255, 255, 0, 0, 255, 255, 0, 128, 203, 42, 128, 128, 0, 0]
    })

# Load YOLO model
try:
    model = YOLO("yolov8n.pt")
    print("✅ YOLO model loaded successfully")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None

# Global variables for statistics and color tracking
detection_stats = {
    'total_detections': 0,
    'color_detections': 0,
    'recent_colors': []
}

# Global variable to store the selected point for continuous color tracking
selected_point = {
    'x': None,
    'y': None,
    'color_info': None,
    'active': False
}

# CRITICAL FIX: Global variable to store the current frame
current_frame = {
    'frame': None,
    'processed_frame': None
}

def get_color_name(R, G, B):
    """Get color name from RGB values"""
    if R < 15 and G < 15 and B < 15:
        return "Very Dark (Near Black)"
    
    minimum = float('inf')
    cname = ""
    
    for i in range(len(df)):
        d = abs(R - int(df.loc[i, 'R'])) + abs(G - int(df.loc[i, 'G'])) + abs(B - int(df.loc[i, 'B']))
        if d <= minimum:
            minimum = d
            cname = df.loc[i, 'color_name']
    
    return cname

def draw_crosshair_and_color_info(frame, x, y, color_info, show_color_detection):
    """Draw crosshair and color information on the frame"""
    if not show_color_detection or x is None or y is None:
        return frame
    
    # Draw crosshair
    crosshair_size = 20
    crosshair_color = (0, 255, 0)  # Green crosshair
    thickness = 2
    
    # Horizontal line
    cv2.line(frame, (x - crosshair_size, y), (x + crosshair_size, y), crosshair_color, thickness)
    # Vertical line
    cv2.line(frame, (x, y - crosshair_size), (x, y + crosshair_size), crosshair_color, thickness)
    
    # Draw center dot
    cv2.circle(frame, (x, y), 3, crosshair_color, -1)
    
    # If we have color info, display it
    if color_info:
        # Parse color info to get RGB values
        try:
            rgb_part = color_info.split(' - ')[0]  # Get "RGB(r, g, b)" part
            rgb_values = rgb_part.replace('RGB(', '').replace(')', '').split(', ')
            r, g, b = int(rgb_values[0]), int(rgb_values[1]), int(rgb_values[2])
            color_name = color_info.split(' - ')[1]  # Get color name part
            
            # Calculate position for color box (avoid edges)
            box_x = max(20, min(x - 100, frame.shape[1] - 220))
            box_y = max(80, y - 60) if y > 100 else y + 40
            
            # Draw color rectangle
            cv2.rectangle(frame, (box_x, box_y - 30), (box_x + 200, box_y + 20), (b, g, r), -1)
            
            # Choose text color based on background brightness
            text_color = (255, 255, 255) if r + g + b < 400 else (0, 0, 0)
            
            # Draw text
            cv2.putText(frame, color_name, (box_x + 5, box_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"RGB({r}, {g}, {b})", (box_x + 5, box_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            
        except:
            pass  # If parsing fails, just skip the color display
    
    return frame

def get_color_at_point(frame_bgr, x, y):
    """Get color information at specific point"""
    if 0 <= y < frame_bgr.shape[0] and 0 <= x < frame_bgr.shape[1]:
        # Get BGR values from frame
        b, g, r = frame_bgr[y, x]
        b, g, r = int(b), int(g), int(r)
        
        # Get color name
        cname = get_color_name(r, g, b)
        
        return f"RGB({r}, {g}, {b}) - {cname}", (r, g, b, cname)
    return None, None

def create_color_palette(colors_list):
    """Create a color palette visualization"""
    if not colors_list:
        return None
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    
    # Create color rectangles
    colors_to_show = colors_list[-10:] if len(colors_list) > 10 else colors_list
    for i, color_info in enumerate(colors_to_show):
        color_name, r, g, b = color_info
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=(r/255, g/255, b/255))
        ax.add_patch(rect)
        ax.text(i+0.5, 0.5, color_name, ha='center', va='center', 
                fontsize=8, rotation=45, color='white' if r+g+b < 384 else 'black')
    
    ax.set_xlim(0, len(colors_to_show))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Recent Color Detections', fontsize=12, pad=20)
    
    plt.tight_layout()
    return fig

def process_frame_streaming(frame, show_edges=False, show_color_detection=True, confidence_threshold=0.4):
    """Process frame for streaming (without click detection)"""
    if frame is None:
        print("❌ No frame in streaming!")
        return None, "", None, get_stats_text()
    
    print("✅ Frame received in streaming!")
    
    # CRITICAL FIX: Store the current frame globally
    current_frame['frame'] = frame
    
    # Convert to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Object detection
    annotated_frame = frame_bgr.copy()
    
    if model is not None:
        try:
            results = model.predict(source=frame_bgr, conf=confidence_threshold, save=False, verbose=False)
            annotated_frame = results[0].plot()
            
            # Update detection stats
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            detection_stats['total_detections'] += num_detections
            
        except Exception as e:
            print(f"Error in object detection: {e}")
            annotated_frame = frame_bgr.copy()
    
    # If we have a selected point, get current color at that point and draw crosshair
    current_color_info = ""
    if selected_point['active'] and show_color_detection:
        color_info, color_data = get_color_at_point(frame_bgr, selected_point['x'], selected_point['y'])
        if color_info:
            selected_point['color_info'] = color_info
            current_color_info = color_info
            
        # Draw crosshair and color info on the annotated frame
        annotated_frame = draw_crosshair_and_color_info(
            annotated_frame, 
            selected_point['x'], 
            selected_point['y'], 
            selected_point['color_info'],
            show_color_detection
        )
    
    # Add edge detection if toggled
    if show_edges:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # If we have a selected point, also draw crosshair on edges
        if selected_point['active'] and show_color_detection:
            edges_bgr = draw_crosshair_and_color_info(
                edges_bgr, 
                selected_point['x'], 
                selected_point['y'], 
                None,  # Don't show color info on edges
                show_color_detection
            )
        
        # Resize both images to same height
        h1, h2 = annotated_frame.shape[0], edges_bgr.shape[0]
        if h1 != h2:
            edges_bgr = cv2.resize(edges_bgr, (edges_bgr.shape[1], h1))
        
        final = np.hstack((annotated_frame, edges_bgr))
    else:
        final = annotated_frame
    
    # Convert back to RGB
    final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    
    # CRITICAL FIX: Store the processed frame globally too
    current_frame['processed_frame'] = final_rgb
    
    # Create color palette only if color detection is enabled
    palette_fig = create_color_palette(detection_stats['recent_colors']) if show_color_detection else None
    
    return final_rgb, current_color_info, palette_fig, get_stats_text()

def handle_click(evt: gr.SelectData, show_edges=False, show_color_detection=True, confidence_threshold=0.4):
    """Handle click events on the image - FIXED VERSION"""
    print(f"🖱️ CLICK EVENT TRIGGERED!")
    print(f"🔍 Click detected at: ({evt.index})")
    print(f"🎨 Color detection enabled: {show_color_detection}")
    
    # CRITICAL FIX: Use the globally stored frame
    frame = current_frame.get('frame')
    print(f"📸 Frame from global storage: {frame is not None}")
    
    if frame is None:
        print("❌ No frame in global storage!")
        # Try to return the last processed frame if available
        if current_frame.get('processed_frame') is not None:
            return current_frame['processed_frame'], "No current frame, but click registered", None, get_stats_text()
        return None, "No frame available for color detection", None, get_stats_text()
    
    print(f"📸 Frame shape: {frame.shape}")
    
    # Get click coordinates
    x, y = evt.index
    print(f"🎯 Processing click at: x={x}, y={y}")
    
    # Update selected point for continuous tracking
    if show_color_detection:
        selected_point['x'] = x
        selected_point['y'] = y
        selected_point['active'] = True
        print(f"✅ Selected point updated: {selected_point}")
    else:
        selected_point['active'] = False
        print("❌ Color detection is disabled")
    
    # Convert to BGR for OpenCV processing
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    print(f"🔄 Frame converted to BGR, shape: {frame_bgr.shape}")
    
    # Object detection first
    annotated_frame = frame_bgr.copy()
    
    if model is not None:
        try:
            results = model.predict(source=frame_bgr, conf=confidence_threshold, save=False, verbose=False)
            annotated_frame = results[0].plot()
        except Exception as e:
            print(f"❌ Error in object detection: {e}")
    
    # Color detection at click position (only if color detection is enabled)
    color_info = ""
    if show_color_detection:
        print(f"🎨 Getting color at point ({x}, {y})")
        color_info, color_data = get_color_at_point(frame_bgr, x, y)
        print(f"🎨 Color info: {color_info}")
        print(f"🎨 Color data: {color_data}")
        
        if color_info and color_data:
            r, g, b, cname = color_data
            
            # Update color statistics
            detection_stats['color_detections'] += 1
            detection_stats['recent_colors'].append((cname, r, g, b))
            
            # Keep only last 20 colors
            if len(detection_stats['recent_colors']) > 20:
                detection_stats['recent_colors'] = detection_stats['recent_colors'][-20:]
            
            print(f"✅ Color detected and saved: {color_info}")
            
            # Store color info for continuous display
            selected_point['color_info'] = color_info
        else:
            print("❌ No color info returned")
        
        # Draw crosshair and color info
        annotated_frame = draw_crosshair_and_color_info(
            annotated_frame, x, y, color_info, show_color_detection
        )
        print("✅ Crosshair drawn")
    else:
        color_info = "Color detection is disabled"
        # Still draw a red dot to show where user clicked
        cv2.circle(annotated_frame, (x, y), 5, (0, 0, 255), -1)  # Red dot for disabled mode
        print("🔴 Red dot drawn (color detection disabled)")
    
    # Add edge detection if toggled
    if show_edges:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # If color detection is enabled, also draw crosshair on edges
        if show_color_detection and selected_point['active']:
            edges_bgr = draw_crosshair_and_color_info(
                edges_bgr, x, y, None, show_color_detection  # Don't show color info on edges
            )
        elif not show_color_detection:
            cv2.circle(edges_bgr, (x, y), 5, (0, 0, 255), -1)  # Red dot for disabled mode
        
        # Resize both images to same height
        h1, h2 = annotated_frame.shape[0], edges_bgr.shape[0]
        if h1 != h2:
            edges_bgr = cv2.resize(edges_bgr, (edges_bgr.shape[1], h1))
        
        final = np.hstack((annotated_frame, edges_bgr))
    else:
        final = annotated_frame
    
    # Convert back to RGB
    final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    print("🔄 Final frame converted back to RGB")
    
    # Update the global processed frame
    current_frame['processed_frame'] = final_rgb
    
    # Create color palette only if color detection is enabled
    palette_fig = create_color_palette(detection_stats['recent_colors']) if show_color_detection else None
    
    print(f"📤 Returning: color_info='{color_info}'")
    return final_rgb, color_info, palette_fig, get_stats_text()

def get_stats_text():
    """Get formatted statistics text"""
    return f"""📊 **Detection Statistics**
• Total Object Detections: {detection_stats['total_detections']}
• Color Detections: {detection_stats['color_detections']}
• Recent Colors Found: {len(detection_stats['recent_colors'])}"""

def reset_stats():
    """Reset detection statistics"""
    global detection_stats, selected_point
    detection_stats = {
        'total_detections': 0,
        'color_detections': 0,
        'recent_colors': []
    }
    # Also reset the selected point
    selected_point = {
        'x': None,
        'y': None,
        'color_info': None,
        'active': False
    }
    return "Statistics reset successfully! Click tracking cleared.", None, get_stats_text()

def clear_selection():
    """Clear the current color selection"""
    global selected_point
    selected_point = {
        'x': None,
        'y': None,
        'color_info': None,
        'active': False
    }
    return "Selection cleared! Click on the camera to select a new area."

def save_color_info():
    """Save recent color detections to a file"""
    if not detection_stats['recent_colors']:
        return "No color data to save!"
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"color_detections_{timestamp}.csv"
    
    # Create DataFrame
    color_data = pd.DataFrame(detection_stats['recent_colors'], 
                            columns=['Color_Name', 'R', 'G', 'B'])
    
    # Save to CSV
    color_data.to_csv(filename, index=False)
    return f"Color data saved to {filename}!"

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.main-header {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}

.stats-box {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}

.color-info {
    font-size: 16px;
    font-weight: bold;
    padding: 10px;
    border-radius: 5px;
    background-color: #e9ecef;
}
"""

# Create Gradio Interface
with gr.Blocks(css=custom_css, title="YOLO Color Detection System") as demo:
    
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🎨 Advanced YOLO + Color Detection System</h1>
        <p>Real-time object detection with intelligent color analysis</p>
    </div>
    """)
    
    # Main interface
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📷 **Camera Feed & Detection**")
            gr.Markdown("**Click anywhere on the image to detect color at that point (when color detection is enabled)**")
            cam = gr.Image(
                sources=["webcam"], 
                streaming=True,
                label="Camera Feed"
            )
            
        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ **Processed Output**")
            output = gr.Image(label="Detection Results")
    
    # Controls and Information
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ⚙️ **Controls**")
            edge_toggle = gr.Checkbox(label="🔍 Show Edge Detection", value=False)
            color_toggle = gr.Checkbox(label="🎨 Show Color Detection", value=True)
            confidence_slider = gr.Slider(
                minimum=0.1, 
                maximum=1.0, 
                value=0.4, 
                step=0.1,
                label="🎯 Detection Confidence Threshold"
            )
            
            with gr.Row():
                reset_btn = gr.Button("🔄 Reset Statistics", variant="secondary")
                save_btn = gr.Button("💾 Save Color Data", variant="primary")
                clear_selection_btn = gr.Button("❌ Clear Selection", variant="secondary")
        
        with gr.Column():
            gr.Markdown("### 🎨 **Color Information**")
            color_text = gr.Textbox(
                label="Detected Color", 
                placeholder="Enable color detection and click on the image to detect colors",
                elem_classes="color-info"
            )
            
            status_text = gr.TextArea(
                label="Status Messages", 
                value="System ready! Enable color detection and click on the camera feed to start color detection.",
                lines=3
            )
    
    # Statistics and Visualization
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 **Detection Statistics**")
            stats_display = gr.Markdown(get_stats_text())
            
        with gr.Column():
            gr.Markdown("### 🎨 **Color Palette**")
            color_palette = gr.Plot(label="Recent Colors")
    
    # Instructions
    with gr.Row():
        gr.Markdown("""
        ### 📖 **Instructions**
        1. **Camera Setup**: Allow camera access when prompted
        2. **Object Detection**: Objects will be automatically detected and highlighted
        3. **Color Detection**: Toggle "Show Color Detection" ON, then **CLICK** anywhere on the camera image to select an area
        4. **Persistent Tracking**: Once you click, a **green crosshair (➕)** will appear and continuously track the color at that location
        5. **Color Display**: The color name and RGB values will be displayed in a floating box near the crosshair
        6. **Clear Selection**: Click "Clear Selection" to remove the crosshair and stop color tracking
        7. **Edge Detection**: Toggle edge detection to see object boundaries
        8. **Adjust Confidence**: Use the slider to change object detection sensitivity
        9. **Save Data**: Click 'Save Color Data' to export detected colors to CSV
        
        ### 🎯 **Features**
        - Real-time YOLO object detection
        - **Persistent crosshair tracking** - Click once to continuously monitor color at that spot
        - **Visual crosshair indicator (➕)** - Shows exactly where you're tracking
        - **Floating color display** - Color name and RGB values appear near the crosshair
        - Accurate color identification from extensive color database
        - Edge detection for better object visualization
        - Statistics tracking and color palette visualization
        - Export functionality for detected colors
        
        ### ⚠️ **Important**: 
        - **Toggle "Show Color Detection" ON** to enable color detection
        - **Click once on any area** (like hair, clothes, objects) to start tracking that color
        - **Green crosshair (➕)** will appear and stay on the selected area
        - **Color updates in real-time** as the camera moves or lighting changes
        - **Click "Clear Selection"** to remove the crosshair and select a new area
        - **Multiple clicks** will move the crosshair to new locations
        """)
    
    # Event handlers
    
    # Handle streaming (continuous object detection)
    cam.stream(
        fn=process_frame_streaming,
        inputs=[cam, edge_toggle, color_toggle, confidence_slider],
        outputs=[output, color_text, color_palette, stats_display],
        stream_every=0.5  # Update every 0.5 seconds to reduce load
    )
    
    # FIXED: Handle clicks (color detection) - Only handle clicks on output image
    output.select(
        fn=handle_click,
        inputs=[edge_toggle, color_toggle, confidence_slider],  # Removed cam from inputs
        outputs=[output, color_text, color_palette, stats_display]
    )
    
    # Handle control changes
    edge_toggle.change(
        fn=process_frame_streaming,
        inputs=[cam, edge_toggle, color_toggle, confidence_slider],
        outputs=[output, color_text, color_palette, stats_display]
    )
    
    color_toggle.change(
        fn=process_frame_streaming,
        inputs=[cam, edge_toggle, color_toggle, confidence_slider],
        outputs=[output, color_text, color_palette, stats_display]
    )
    
    confidence_slider.change(
        fn=process_frame_streaming,
        inputs=[cam, edge_toggle, color_toggle, confidence_slider],
        outputs=[output, color_text, color_palette, stats_display]
    )
    
    # Handle button clicks
    reset_btn.click(
        fn=reset_stats,
        outputs=[status_text, color_palette, stats_display]
    )
    
    save_btn.click(
        fn=save_color_info,
        outputs=[status_text]
    )
    
    clear_selection_btn.click(
        fn=clear_selection,
        outputs=[status_text]
    )

# Launch the application
if __name__ == "__main__":
    print("🚀 Starting YOLO Color Detection System...")
    print("📋 Make sure you have:")
    print("   - colors.csv file in the same directory (optional - will create basic one)")
    print("   - yolov8n.pt model file")
    print("   - Camera connected and accessible")
    print("\n🌐 Launching web interface...")
    print("\n⚠️  IMPORTANT: Toggle 'Show Color Detection' ON and then click on the PROCESSED OUTPUT image (right side) to detect colors!")
    
    demo.launch(
        share=False,  # Set to True if you want to create a public link
        debug=True,   # Enable debug mode to see click events
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )