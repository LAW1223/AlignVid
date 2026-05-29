"""
I2V Evaluation Script

Evaluates video quality across 5 dimensions:
- subject_consistency: Subject consistency across frames
- background_consistency: Background consistency across frames  
- motion_smoothness: Smoothness of motion transitions
- dynamic_degree: Amount of dynamic content/motion
- aesthetic_quality: Overall visual aesthetic quality

USAGE:
------
Basic usage:
python evaluate_i2v.py --videos_path /path/to/videos --output_path /path/to/results

Full example with all options:
python evaluate_i2v.py \
    --videos_path /home/user/videos \
    --output_path /home/user/results \
    --temp_folder /tmp/resized_videos \
    --force_ratio "16-9" \
    --keep_temp

ARGUMENTS:
----------
--videos_path       : (Required) Path to folder containing videos to evaluate
--output_path       : Output directory for results (default: ./evaluation_i2v_results/)
--temp_folder       : Temporary folder for resized videos (default: ./temp_resized_videos/)
--force_ratio       : Force specific aspect ratio (e.g., "16-9", "1-1", "4-3")
                      If not specified, automatically detects nearest common ratio
--keep_temp         : Keep temporary resized videos after evaluation
--full_json_dir     : Path to VBench JSON config file

EXAMPLES:
---------
# Evaluate videos with auto-detected aspect ratio
python evaluate_i2v.py --videos_path ./my_videos --output_path ./results

# Force 16:9 aspect ratio and keep temporary files
python evaluate_i2v.py \
    --videos_path ./videos \
    --output_path ./results \
    --force_ratio "16-9" \
    --keep_temp

# Use custom temporary folder
python evaluate_i2v.py \
    --videos_path /data/videos \
    --output_path /data/results \
    --temp_folder /tmp/video_processing

OUTPUT:
-------
Creates JSON file with results in format:
{
    "scores": {
        "dimension_name": [
            avg_score,           // Average score
            video_count,         // Video count
            [                    // Score array for each video
                [video_name, score],
                [video_name, score]
            ]
        ]
    },
    "total_score": 0.901,    // Average score of 5 dimensions
    "video_count": 2         // Total video count
}

REQUIREMENTS:
-------------
- CUDA-capable GPU recommended
- Models will be cached in ~/.cache/vbench/ (~1.7GB total)
- Videos will be resized to match target aspect ratio
- Supported formats: .mp4, .avi, .mov, .mkv, .flv, .wmv
"""

import torch
import os
import cv2
import numpy as np
from vbench2_beta_i2v import VBenchI2V
from datetime import datetime
import argparse
import json
from pathlib import Path
from tqdm import tqdm

# VBench I2V Constants for scoring (from constant.py)
I2VKEY = {
    "camera_motion": "Video-Text Camera Motion",
    "i2v_subject": "Video-Image Subject Consistency",
    "i2v_background": "Video-Image Background Consistency",
    "subject_consistency": "Subject Consistency",
    "background_consistency": "Background Consistency",
    "motion_smoothness": "Motion Smoothness",
    "dynamic_degree": "Dynamic Degree",
    "aesthetic_quality": "Aesthetic Quality",
    "imaging_quality": "Imaging Quality",
}





class ProgressVBenchI2V(VBenchI2V):
    """Enhanced VBenchI2V with progress tracking"""
    
    def evaluate(self, videos_path, name, dimension_list=None, custom_image_folder=None, mode='vbench_standard', local=False, read_frame=False, resolution="1-1", **kwargs):
        results_dict = {}
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        
        print(f"\n🔧 Initializing evaluation modules...")
        submodules_dict = self.init_submodules_with_progress(dimension_list, local=local, read_frame=read_frame, resolution=resolution)
        
        print(f"📋 Building evaluation dataset...")
        cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list, custom_image_folder=custom_image_folder, mode=mode)
        
        print(f"\n🎯 Starting evaluation on {len(dimension_list)} dimensions...")
        
        # Create progress bar for dimensions
        dim_pbar = tqdm(dimension_list, 
                       desc="📊 Evaluating Dimensions", 
                       unit="dims",
                       position=0)
        
        for dimension in dim_pbar:
            dim_pbar.set_description(f"📊 Evaluating: {dimension}")
            
            try:
                if dimension in self.i2v_dims:
                    dimension_module = __import__(f'vbench2_beta_i2v.{dimension}', fromlist=[dimension])
                else:
                    dimension_module = __import__(f'vbench.{dimension}', fromlist=[dimension])
                evaluate_func = getattr(dimension_module, f'compute_{dimension}')
            except Exception as e:
                dim_pbar.write(f"❌ Failed to load dimension {dimension}: {e}")
                raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
            
            submodules_list = submodules_dict[dimension]
            dim_pbar.write(f"🔍 Computing {dimension}...")
            
            try:
                results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
                results_dict[dimension] = results
                dim_pbar.write(f"✅ Completed {dimension}")
            except Exception as e:
                dim_pbar.write(f"❌ Error in {dimension}: {str(e)}")
                # Continue with other dimensions instead of stopping
                dim_pbar.write(f"⚠️  Skipping {dimension} and continuing with remaining dimensions...")
                continue
        
        dim_pbar.close()
        
        # Calculate scores after all evaluations complete
        print(f"\n🧮 Calculating final scores...")
        scored_results = self.calculate_scores(results_dict)
        
        # Only save the main eval_results.json file
        output_name = os.path.join(self.output_path, name+'_eval_results.json')
        self.save_json(scored_results, output_name)
        print(f'\n🎉 Evaluation completed! Results saved to: {output_name}')
        
        # Clean up the temporary full_info.json file
        try:
            if os.path.exists(cur_full_info_path):
                os.remove(cur_full_info_path)
                print(f"🧹 Cleaned up temporary file: {cur_full_info_path}")
        except Exception as e:
            print(f"⚠️  Could not clean up {cur_full_info_path}: {e}")
        
        # Print score summary
        self.print_score_summary(scored_results)
        
        return scored_results
    
    def init_submodules_with_progress(self, dimension_list, local=False, read_frame=False, resolution="1-1"):
        """Initialize submodules with progress tracking"""
        from vbench2_beta_i2v.utils import init_submodules
        
        print("⚙️  Initializing models (this may take a while for first run)...")
        
        # Show which models will be loaded
        model_info = {
            'i2v_subject': 'DINO ViT-B/16 (~330MB)',
            'subject_consistency': 'DINO ViT-B/16 (~330MB)', 
            'background_consistency': 'CLIP ViT-B/32 (~338MB)',
            'aesthetic_quality': 'CLIP ViT-L/14 (~890MB)',
            'motion_smoothness': 'AMT Model (~45MB)',
            'dynamic_degree': 'RAFT Model (~45MB)',
            'imaging_quality': 'MUSIQ Model (~50MB)',
            'camera_motion': 'Co-Tracker Model'
        }
        
        models_to_load = [dim for dim in dimension_list if dim in model_info]
        if models_to_load:
            print("📦 Models to be loaded:")
            for dim in models_to_load:
                print(f"   • {dim}: {model_info[dim]}")
        
        return init_submodules(dimension_list, local=local, read_frame=read_frame, resolution=resolution)
    
    
    def calculate_scores(self, results_dict):
        """Calculate scores for the 5 required dimensions with compact structure"""
        print("📊 Processing dimension results...")
        
        # Required dimensions for this evaluation
        required_dimensions = [
            "subject_consistency",
            "background_consistency", 
            "motion_smoothness",
            "dynamic_degree",
            "aesthetic_quality"
        ]
        
        # Compact results structure
        results = {
            "scores": {},
            "total_score": 0.0,
            "video_count": 0
        }
        
        # Process each dimension's results
        for dimension, raw_results in results_dict.items():
            if dimension not in required_dimensions:
                continue
                
            dimension_key = I2VKEY.get(dimension, dimension)
            
            if isinstance(raw_results, (list, tuple)) and len(raw_results) == 2:
                # VBench format: [average_score, [individual_video_results]]
                avg_score = raw_results[0]
                video_results_list = raw_results[1]
                
                if video_results_list:
                    video_scores = []
                    
                    for item in video_results_list:
                        if isinstance(item, dict) and 'video_results' in item:
                            raw_score = item['video_results']
                            # Handle boolean results (e.g., dynamic_degree)
                            if isinstance(raw_score, bool):
                                raw_score = 1.0 if raw_score else 0.0
                            elif not isinstance(raw_score, (int, float)):
                                continue
                                
                            video_name = item.get('video_path', 'unknown').split('/')[-1]
                            
                            # Video score format
                            video_scores.append([video_name, raw_score])
                    
                    if video_scores:
                        # Calculate average
                        scores = [v[1] for v in video_scores]
                        avg_score = sum(scores) / len(scores)
                        
                        # Dimension format: [avg_score, video_count, [[name, score], ...]]
                        results["scores"][dimension] = [
                            round(avg_score, 4),
                            len(video_scores),
                            video_scores
                        ]
                        
                        # Update video count (use max across dimensions)
                        results["video_count"] = max(results["video_count"], len(video_scores))
                        
                        print(f"   • {dimension}: {avg_score:.4f} ({len(video_scores)} videos)")
                    else:
                        print(f"   ⚠️  {dimension}: No valid video scores found")
                else:
                    print(f"   ⚠️  {dimension}: No video results list")
                    
            elif isinstance(raw_results, dict) and 'video_results' in raw_results:
                # Standard format: {'video_results': [...]}
                video_results = raw_results['video_results']
                if video_results:
                    video_scores = []
                    
                    for item in video_results:
                        if 'score' in item:
                            score = item['score']
                            video_name = item.get('video_list', [item.get('video', 'unknown')])[0] if isinstance(item.get('video_list'), list) else item.get('video', 'unknown')
                            video_scores.append([video_name, score])
                    
                    if video_scores:
                        scores = [v[1] for v in video_scores]
                        avg_score = sum(scores) / len(scores)
                        
                        results["scores"][dimension] = [
                            round(avg_score, 4),
                            len(video_scores),
                            video_scores
                        ]
                        
                        results["video_count"] = max(results["video_count"], len(video_scores))
                        print(f"   • {dimension}: {avg_score:.4f} ({len(video_scores)} videos)")
                    else:
                        print(f"   ⚠️  {dimension}: No valid scores found")
                else:
                    print(f"   ⚠️  {dimension}: No video results")
                    
            elif isinstance(raw_results, (int, float)):
                # Direct score value
                results["scores"][dimension] = [
                    round(raw_results, 4),
                    1,
                    [["single_result", raw_results]]
                ]
                results["video_count"] = max(results["video_count"], 1)
                print(f"   • {dimension}: {raw_results:.4f}")
            else:
                print(f"   ⚠️  {dimension}: Unexpected result format: {type(raw_results)}")
        
        # Calculate Total Score (simple average of all 5 dimensions)
        valid_scores = []
        for dim in required_dimensions:
            if dim in results["scores"]:
                valid_scores.append(results["scores"][dim][0])  # average score
        
        if valid_scores:
            results["total_score"] = round(sum(valid_scores) / len(valid_scores), 4)
            print(f"   🎯 Total Score: {results['total_score']:.4f} (average of {len(valid_scores)} dimensions)")
        else:
            results["total_score"] = 0.0
            print(f"   ⚠️  Total Score: No valid dimensions found")
        
        return results
    

    
    def print_score_summary(self, results):
        """Print a formatted summary of the evaluation scores"""
        print("\n" + "=" * 60)
        print("📊 EVALUATION SCORE SUMMARY - 5 DIMENSIONS")
        print("=" * 60)
        
        total_score = results.get('total_score', 0)
        video_count = results.get('video_count', 0)
        print(f"🏆 Total Score: {total_score:.4f} ({video_count} videos)")
        
        print("\n📋 Dimension Breakdown:")
        scores = results.get('scores', {})
        
        required_dims = [
            "subject_consistency",
            "background_consistency", 
            "motion_smoothness",
            "dynamic_degree",
            "aesthetic_quality"
        ]
        
        for dim in required_dims:
            if dim in scores:
                # scores[dim] format: [avg_score, video_count, video_scores]
                avg_score, dim_video_count, video_scores = scores[dim]
                print(f"   • {dim}: {avg_score:.4f} ({dim_video_count} videos)")
                
                # Show individual video scores if there are multiple videos
                if dim_video_count > 1:
                    for video_name, score in video_scores:
                        print(f"     - {video_name}: {score:.4f}")
            else:
                print(f"   ❌ {dim}: Not evaluated")
        
        print("=" * 60)
    
    def save_json(self, data, path, indent=4):
        """Save JSON with proper formatting"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

def get_video_resolution(video_path):
    """Extract video resolution (width, height) from video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def find_nearest_integer_ratio(width, height):
    """Find the nearest common integer aspect ratio for given dimensions"""
    actual_ratio = width / height
    
    # Common integer aspect ratios (1:1 to 16:9 range)
    common_ratios = [
        (1, 1),    # 1:1 = 1.0
        (5, 4),    # 5:4 = 1.25
        (4, 3),    # 4:3 = 1.333
        (3, 2),    # 3:2 = 1.5
        (8, 5),    # 8:5 = 1.6
        (5, 3),    # 5:3 = 1.667
        (16, 9),   # 16:9 = 1.778
    ]
    
    # Find the closest ratio
    best_ratio = None
    min_diff = float('inf')
    
    for w_ratio, h_ratio in common_ratios:
        ratio_value = w_ratio / h_ratio
        diff = abs(actual_ratio - ratio_value)
        if diff < min_diff:
            min_diff = diff
            best_ratio = (w_ratio, h_ratio)
    
    return best_ratio, f"{best_ratio[0]}-{best_ratio[1]}"

def resize_video(input_path, output_path, target_width, target_height, pbar=None):
    """Resize video to target resolution while maintaining frame rate"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open video: {input_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    # Create frame progress bar
    frame_pbar = tqdm(total=total_frames, 
                     desc=f"Processing {os.path.basename(input_path)}", 
                     unit="frames", 
                     leave=False,
                     position=1)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        resized_frame = cv2.resize(frame, (target_width, target_height))
        out.write(resized_frame)
        frame_count += 1
        frame_pbar.update(1)
    
    frame_pbar.close()
    cap.release()
    out.release()
    
    if pbar:
        pbar.set_postfix_str(f"Completed: {os.path.basename(input_path)}")
    
    return True

def process_videos_in_folder(videos_folder, temp_folder, target_ratio_str):
    """Process all videos in folder to match target aspect ratio"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    processed_videos = []
    
    target_w, target_h = map(int, target_ratio_str.split('-'))
    
    # First, collect all video files
    video_files = []
    for file_path in Path(videos_folder).rglob('*'):
        if file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    if not video_files:
        print("No video files found to process")
        return processed_videos
    
    print(f"\n📹 Processing {len(video_files)} videos...")
    
    # Create main progress bar for video processing
    main_pbar = tqdm(video_files, 
                     desc="🎬 Resizing Videos", 
                     unit="videos",
                     position=0)
    
    for file_path in main_pbar:
        main_pbar.set_description(f"🎬 Processing: {file_path.name}")
        
        width, height = get_video_resolution(str(file_path))
        if width is None or height is None:
            main_pbar.write(f"⚠️  Skipping invalid video: {file_path.name}")
            continue
        
        # Calculate target resolution (maintain longer side, adjust shorter side)
        if width >= height:
            # Landscape video
            new_width = max(width, 512)  # Minimum 512 pixels
            new_height = int(new_width * target_h / target_w)
        else:
            # Portrait video
            new_height = max(height, 512)  # Minimum 512 pixels
            new_width = int(new_height * target_w / target_h)
        
        # Generate output path
        relative_path = file_path.relative_to(videos_folder)
        output_path = Path(temp_folder) / relative_path
        
        # Resize video with progress tracking
        if resize_video(str(file_path), str(output_path), new_width, new_height, main_pbar):
            processed_videos.append(str(output_path))
            main_pbar.write(f"✅ Completed: {file_path.name} -> {new_width}x{new_height}")
        else:
            main_pbar.write(f"❌ Failed: {file_path.name}")
    
    main_pbar.close()
    print(f"\n🎉 Successfully processed {len(processed_videos)}/{len(video_files)} videos")
    
    return processed_videos

def parse_args():
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='VBenchI2V - Enhanced Video Evaluation Framework')
    parser.add_argument(
        "--videos_path",
        type=str,
        required=True,
        help="Path to folder containing videos to evaluate",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='./evaluation_i2v_results/',
        help="Output path for saving evaluation results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default='results',
        help="Name of the evaluation results",
    )
    parser.add_argument(
        "--full_json_dir",
        type=str,
        default=f'{CUR_DIR}/vbench2_beta_i2v/vbench2_i2v_full_info.json',
        help="Path to JSON file containing prompt and dimension information",
    )
    parser.add_argument(
        "--temp_folder",
        type=str,
        default='./temp_resized_videos/',
        help="Temporary folder for storing resized videos",
    )
    parser.add_argument(
        "--keep_temp",
        action='store_true',
        help="Keep temporary resized videos after evaluation",
    )
    parser.add_argument(
        "--force_ratio",
        type=str,
        default=None,
        help="Force specific aspect ratio (e.g., '16-9'), otherwise auto-detect nearest ratio",
    )
    parser.add_argument(
        "--dimension",
        nargs="+",
        default=["dynamic_degree", "aesthetic_quality"],
        help="VBench dimensions to evaluate (default: the two reported in the paper)",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(f'Arguments: {args}')
    
    # Validate input folder
    if not os.path.exists(args.videos_path):
        print(f"Error: Video folder does not exist: {args.videos_path}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    print(f"📁 Output directory ready: {args.output_path}")
    
    # Dimensions to evaluate. Default is Dynamic Degree + Aesthetic Quality
    # (the two visual-quality metrics reported in the paper); override with --dimension.
    dimensions = args.dimension
    
    print(f"Evaluation dimensions: {dimensions}")
    
    # Discover video files
    video_files = []
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    for file_path in Path(args.videos_path).rglob('*'):
        if file_path.suffix.lower() in video_extensions:
            video_files.append(str(file_path))
    
    if not video_files:
        print(f"Error: No video files found in {args.videos_path}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Determine target aspect ratio
    if args.force_ratio:
        target_ratio_str = args.force_ratio
        print(f"Using forced aspect ratio: {target_ratio_str}")
    else:
        # Analyze first video to determine target ratio
        first_video = video_files[0]
        width, height = get_video_resolution(first_video)
        if width is None or height is None:
            print(f"Error: Cannot read video resolution: {first_video}")
            return
        
        nearest_ratio, target_ratio_str = find_nearest_integer_ratio(width, height)
        print(f"Detected video resolution: {width}x{height} (ratio: {width/height:.3f})")
        print(f"Selected nearest integer ratio: {nearest_ratio[0]}:{nearest_ratio[1]} ({target_ratio_str})")
    
    # Process videos - resize to target ratio
    print("Starting video processing...")
    processed_videos = process_videos_in_folder(args.videos_path, args.temp_folder, target_ratio_str)
    
    if not processed_videos:
        print("Error: No videos were successfully processed")
        return
    
    print(f"Successfully processed {len(processed_videos)} videos")
    
    # Cache directory for VBench's backbone model downloads (override with VBENCH_CACHE_DIR).
    cache_dir = os.environ.get(
        "VBENCH_CACHE_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".vbench_cache"))
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["VBENCH_CACHE_DIR"] = cache_dir
    
    # Initialize VBench evaluation framework
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    my_VBench = ProgressVBenchI2V(device, args.full_json_dir, args.output_path)
    
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Configuration for image quality preprocessing
    kwargs = {
        'imaging_quality_preprocessing_mode': 'longer'
    }
    
    try:
        print(f"\n🎬 Starting VBench I2V Evaluation")
        print(f"📁 Input: {args.temp_folder}")
        print(f"📊 Dimensions: {', '.join(dimensions)}")
        print(f"🎯 Target ratio: {target_ratio_str}")
        print("=" * 60)
        
        results = my_VBench.evaluate(
            videos_path=args.temp_folder,
            name=f'{args.name}_{current_time}_{target_ratio_str}',
            dimension_list=dimensions,
            resolution=target_ratio_str,
            mode='custom_input',
            **kwargs
        )
        
        print("=" * 60)
        print(f"\n🎊 All done! Check your results in: {args.output_path}")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary files
        if not args.keep_temp:
            import shutil
            if os.path.exists(args.temp_folder):
                print(f"\n🧹 Cleaning up temporary files...")
                shutil.rmtree(args.temp_folder)
                print(f"✅ Cleaned up temporary folder: {args.temp_folder}")
        else:
            print(f"\n📁 Keeping temporary folder: {args.temp_folder}")

if __name__ == "__main__":
    main()

'''
python evaluate_i2v.py --videos_path /path/to/generated_videos --output_path ./results --name my_run
'''