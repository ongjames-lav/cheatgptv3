"""
Report Generator for CheatGPT3 Video Processing
Generates comprehensive reports and visualizations from processed video data
"""

import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np

# Try to import visualization libraries, fall back to basic functionality if not available
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI threading issues
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
    # Set up matplotlib for server environments
    plt.style.use('default')
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (12, 8)
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available, visualization features disabled")


class MatplotlibManager:
    """Context manager to ensure proper matplotlib cleanup and prevent threading issues"""
    
    def __init__(self, figsize=(10, 6)):
        self.figsize = figsize
        self.fig = None
        self.ax = None
        
    def __enter__(self):
        if VISUALIZATION_AVAILABLE:
            plt.ioff()  # Turn off interactive mode
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            return self.fig, self.ax
        return None, None
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if VISUALIZATION_AVAILABLE and self.fig is not None:
            plt.close(self.fig)
            plt.clf()  # Clear the current figure
            plt.cla()  # Clear the current axes


class ReportGenerator:
    """Generates various types of reports from video processing results"""
    
    def __init__(self, output_dir: str = "uploads/reports"):
        self.output_dir = output_dir
        self.ensure_directories()
        
        # Set up matplotlib for server environments if available
        if VISUALIZATION_AVAILABLE:
            try:
                plt.style.use('default')
                plt.rcParams['font.size'] = 10
                plt.rcParams['figure.figsize'] = (12, 8)
            except Exception as e:
                print(f"Warning: matplotlib setup failed: {e}")
    
    def ensure_directories(self):
        """Create output directories"""
        dirs = ['json', 'csv', 'statistics', 'visualizations', 'summaries']
        for dir_name in dirs:
            os.makedirs(os.path.join(self.output_dir, dir_name), exist_ok=True)
    
    def generate_comprehensive_report(self, session_id: str, video_metadata: Dict,
                                    events: List[Dict], summary: Dict) -> Dict[str, str]:
        """
        Generate comprehensive analysis report
        
        Args:
            session_id: Session identifier
            video_metadata: Video file metadata
            events: List of detected events
            summary: Event summary statistics
            
        Returns:
            Dict containing paths to generated report files
        """
        report_paths = {}
        
        try:
            # Generate JSON report
            json_path = self.generate_json_report(session_id, video_metadata, events, summary)
            if json_path:
                report_paths['json_report'] = json_path
            
            # Generate CSV report
            csv_path = self.generate_csv_report(session_id, events)
            if csv_path:
                report_paths['csv_report'] = csv_path
            
            # Generate executive summary
            summary_path = self.generate_executive_summary(session_id, video_metadata, events, summary)
            if summary_path:
                report_paths['executive_summary'] = summary_path
            
            # Generate statistics report
            stats_path = self.generate_statistics_report(session_id, video_metadata, events, summary)
            if stats_path:
                report_paths['statistics_report'] = stats_path
                
        except Exception as e:
            print(f"Error generating reports: {e}")
        
        return report_paths
    
    def generate_json_report(self, session_id: str, video_metadata: Dict,
                           events: List[Dict], summary: Dict) -> Optional[str]:
        """Generate detailed JSON report"""
        try:
            report_data = {
                'session_info': {
                    'session_id': session_id,
                    'generated_at': datetime.now().isoformat(),
                    'generator_version': '3.0.0'
                },
                'video_metadata': video_metadata,
                'analysis_results': {
                    'total_events': len(events),
                    'event_summary': summary,
                    'events': events
                },
                'risk_assessment': {
                    'risk_score': self._calculate_risk_score(summary, video_metadata.get('duration', 0)),
                    'risk_level': self._get_risk_level(self._calculate_risk_score(summary, video_metadata.get('duration', 0))),
                    'recommendations': self._generate_recommendations(summary)
                }
            }
            
            output_path = os.path.join(self.output_dir, 'json', f'report_{session_id}.json')
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            return output_path
            
        except Exception as e:
            print(f"Error generating JSON report: {e}")
            return None
    
    def generate_csv_report(self, session_id: str, events: List[Dict]) -> Optional[str]:
        """Generate CSV report of events"""
        try:
            if not events:
                return None
            
            output_path = os.path.join(self.output_dir, 'csv', f'events_{session_id}.csv')
            
            # Define CSV columns
            columns = [
                'timestamp', 'person_id', 'event_type', 'severity', 'confidence',
                'source', 'details', 'bbox', 'rule_triggered'
            ]
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                
                for event in events:
                    # Flatten event data for CSV
                    row = {}
                    for col in columns:
                        value = event.get(col, '')
                        if isinstance(value, (list, dict)):
                            row[col] = str(value)
                        else:
                            row[col] = value
                    writer.writerow(row)
            
            return output_path
            
        except Exception as e:
            print(f"Error generating CSV report: {e}")
            return None
    
    def generate_visualizations(self, session_id: str, video_metadata: Dict,
                              events: List[Dict], summary: Dict) -> Dict[str, str]:
        """Generate visualization charts"""
        viz_paths = {}
        
        if not VISUALIZATION_AVAILABLE:
            # Create a simple text file explaining no visualizations available
            viz_dir = os.path.join(self.output_dir, 'visualizations')
            readme_path = os.path.join(viz_dir, f'README_{session_id}.txt')
            with open(readme_path, 'w') as f:
                f.write("Visualization libraries not available.\n")
                f.write("Install matplotlib, pandas, and seaborn for visualization features.\n")
                f.write(f"Session: {session_id}\n")
                f.write(f"Events detected: {len(events)}\n")
            viz_paths['info'] = readme_path
            return viz_paths
        
        if not events:
            # Create empty visualization for no events
            with MatplotlibManager(figsize=(10, 6)) as (fig, ax):
                if fig is not None:
                    ax.text(0.5, 0.5, 'No events detected in this video', 
                           ha='center', va='center', fontsize=16)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    
                    viz_dir = os.path.join(self.output_dir, 'visualizations')
                    path = os.path.join(viz_dir, 'no_events.png')
                    plt.savefig(path, dpi=150, bbox_inches='tight')
                    viz_paths['no_events'] = path
            return viz_paths
        
        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        
        # 1. Event Distribution Pie Chart
        try:
            with MatplotlibManager(figsize=(10, 8)) as (fig, ax):
                if fig is not None:
                    event_counts = [count for count in summary.values() if count > 0]
                    event_labels = [event_type.replace('_', ' ').title() 
                                   for event_type, count in summary.items() if count > 0]
                    
                    if event_counts:
                        colors = plt.cm.Set3(np.linspace(0, 1, len(event_counts)))
                        wedges, texts, autotexts = ax.pie(event_counts, labels=event_labels, autopct='%1.1f%%',
                                                         colors=colors, startangle=90)
                        
                        # Beautify text
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_weight('bold')
                        
                        ax.set_title(f'Event Distribution - {session_id}', fontsize=16, fontweight='bold')
                        
                        path = os.path.join(viz_dir, 'event_distribution.png')
                        plt.savefig(path, dpi=150, bbox_inches='tight')
                        viz_paths['event_distribution'] = path
        except Exception as e:
            print(f"Error creating event distribution chart: {e}")
        
        # 2. Timeline Plot
        try:
            if events:
                with MatplotlibManager(figsize=(15, 8)) as (fig, ax):
                    if fig is not None:
                        # Group events by type for different colors
                        event_types = {}
                        for event in events:
                            event_type = event.get('event_type', 'unknown')
                            if event_type not in event_types:
                                event_types[event_type] = []
                            event_types[event_type].append(event.get('timestamp', 0))
                        
                        colors = plt.cm.tab10(np.linspace(0, 1, len(event_types)))
                        
                        for i, (event_type, timestamps) in enumerate(event_types.items()):
                            y_pos = [i] * len(timestamps)
                            ax.scatter(timestamps, y_pos, c=[colors[i]], s=50, alpha=0.7, label=event_type)
                        
                        ax.set_xlabel('Time (seconds)', fontsize=12)
                        ax.set_ylabel('Event Types', fontsize=12)
                        ax.set_title(f'Event Timeline - {session_id}', fontsize=16, fontweight='bold')
                        ax.set_yticks(range(len(event_types)))
                        ax.set_yticklabels(list(event_types.keys()))
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        ax.grid(True, alpha=0.3)
                        
                        path = os.path.join(viz_dir, 'event_timeline.png')
                        plt.savefig(path, dpi=150, bbox_inches='tight')
                        viz_paths['event_timeline'] = path
        except Exception as e:
            print(f"Error creating timeline plot: {e}")
        
        # 3. Confidence Distribution
        try:
            if events:
                confidences = [event.get('confidence', 0) for event in events if event.get('confidence')]
                if confidences:
                    with MatplotlibManager(figsize=(10, 6)) as (fig, ax):
                        if fig is not None:
                            ax.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                            ax.set_xlabel('Confidence Score', fontsize=12)
                            ax.set_ylabel('Frequency', fontsize=12)
                            ax.set_title(f'Detection Confidence Distribution - {session_id}', 
                                       fontsize=16, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            
                            # Add statistics text
                            mean_conf = np.mean(confidences)
                            median_conf = np.median(confidences)
                            ax.axvline(mean_conf, color='red', linestyle='--', 
                                      label=f'Mean: {mean_conf:.3f}')
                            ax.axvline(median_conf, color='orange', linestyle='--', 
                                      label=f'Median: {median_conf:.3f}')
                            ax.legend()
                            
                            path = os.path.join(viz_dir, 'confidence_distribution.png')
                            plt.savefig(path, dpi=150, bbox_inches='tight')
                            viz_paths['confidence_distribution'] = path
        except Exception as e:
            print(f"Error creating confidence distribution: {e}")
        
        # 4. Heat Map of Activity
        try:
            if events and video_metadata.get('duration', 0) > 0:
                duration = video_metadata['duration']
                # Create time bins (e.g., 10-second intervals)
                bin_size = 10  # seconds
                num_bins = int(np.ceil(duration / bin_size))
                
                activity_matrix = np.zeros((len(summary), num_bins))
                event_type_list = list(summary.keys())
                
                for event in events:
                    timestamp = event.get('timestamp', 0)
                    event_type = event.get('event_type', '')
                    
                    bin_idx = min(int(timestamp / bin_size), num_bins - 1)
                    if event_type in event_type_list:
                        type_idx = event_type_list.index(event_type)
                        activity_matrix[type_idx, bin_idx] += 1
                
                if np.sum(activity_matrix) > 0:
                    with MatplotlibManager(figsize=(15, 8)) as (fig, ax):
                        if fig is not None:
                            sns.heatmap(activity_matrix, 
                                       xticklabels=[f'{i*bin_size}s' for i in range(num_bins)],
                                       yticklabels=[t.replace('_', ' ').title() for t in event_type_list],
                                       cmap='YlOrRd', annot=False, ax=ax)
                            
                            ax.set_title(f'Activity Heatmap - {session_id}', fontsize=16, fontweight='bold')
                            ax.set_xlabel('Time', fontsize=12)
                            ax.set_ylabel('Event Types', fontsize=12)
                            
                            path = os.path.join(viz_dir, 'activity_heatmap.png')
                            plt.savefig(path, dpi=150, bbox_inches='tight')
                            viz_paths['activity_heatmap'] = path
        except Exception as e:
            print(f"Error creating activity heatmap: {e}")
        
        return viz_paths
    
    def generate_executive_summary(self, session_id: str, video_metadata: Dict,
                                 events: List[Dict], summary: Dict) -> str:
        """Generate executive summary report"""
        
        risk_score = self._calculate_risk_score(summary, video_metadata.get('duration', 0))
        risk_level = self._get_risk_level(risk_score)
        
        # Create summary text
        summary_text = f"""
CHEATGPT3 VIDEO ANALYSIS EXECUTIVE SUMMARY
==========================================

Session ID: {session_id}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

VIDEO INFORMATION
-----------------
File: {video_metadata.get('filename', 'Unknown')}
Duration: {video_metadata.get('duration', 0):.1f} seconds
Resolution: {video_metadata.get('width', 0)}x{video_metadata.get('height', 0)}
Frame Rate: {video_metadata.get('fps', 0):.1f} FPS

DETECTION SUMMARY
-----------------
Total Events Detected: {len(events)}

Event Breakdown:
"""
        
        for event_type, count in summary.items():
            summary_text += f"  • {event_type.replace('_', ' ').title()}: {count}\n"
        
        summary_text += f"""
RISK ASSESSMENT
--------------
Risk Score: {risk_score:.2f}/10.0
Risk Level: {risk_level.upper()}

RECOMMENDATIONS
--------------
"""
        
        recommendations = self._generate_recommendations(summary)
        for rec in recommendations:
            summary_text += f"  • {rec}\n"
        
        # Save summary
        output_path = os.path.join(self.output_dir, 'summaries', f'executive_summary_{session_id}.txt')
        
        with open(output_path, 'w') as f:
            f.write(summary_text)
        
        return output_path
    
    def generate_statistics_report(self, session_id: str, video_metadata: Dict,
                                 events: List[Dict], summary: Dict) -> str:
        """Generate detailed statistics report"""
        
        stats_text = f"""
CHEATGPT3 DETAILED STATISTICS REPORT
===================================

Session: {session_id}
Generated: {datetime.now().isoformat()}

VIDEO STATISTICS
---------------
Duration: {video_metadata.get('duration', 0):.2f} seconds
Total Frames: {video_metadata.get('frame_count', 0)}
Processing Rate: {video_metadata.get('fps', 0):.1f} FPS

EVENT STATISTICS
---------------
Total Events: {len(events)}
Events per Minute: {(len(events) / max(video_metadata.get('duration', 1), 1)) * 60:.2f}

Event Type Distribution:
"""
        
        total_events = sum(summary.values())
        for event_type, count in sorted(summary.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_events * 100) if total_events > 0 else 0
            stats_text += f"  {event_type.replace('_', ' ').title():.<30} {count:>5} ({percentage:>5.1f}%)\n"
        
        if events:
            confidences = [e.get('confidence', 0) for e in events if e.get('confidence')]
            if confidences:
                stats_text += f"""
CONFIDENCE STATISTICS
--------------------
Mean Confidence: {np.mean(confidences):.3f}
Median Confidence: {np.median(confidences):.3f}
Min Confidence: {np.min(confidences):.3f}
Max Confidence: {np.max(confidences):.3f}
Std Deviation: {np.std(confidences):.3f}
"""
        
        # Save statistics
        output_path = os.path.join(self.output_dir, 'statistics', f'statistics_{session_id}.txt')
        
        with open(output_path, 'w') as f:
            f.write(stats_text)
        
        return output_path
    
    def _calculate_risk_score(self, summary: Dict, duration: float) -> float:
        """Calculate risk score based on detected events"""
        if duration <= 0:
            return 0.0
        
        # Event severity weights (only 3 active detections)
        event_weights = {
            'Phone Usage Detected': 8.0,
            'head_turn_frequent': 4.0,
            'head_turn_sustained': 3.0,
            'hand_extended_duration': 3.0,
            # Disabled detections (keeping for backward compatibility with old data)
            # 'head_pitch_sustained': 2.0,
            # 'out_of_frame_duration': 2.0
        }
        
        total_weighted_score = 0.0
        for event_type, count in summary.items():
            weight = event_weights.get(event_type, 1.0)
            # Normalize by duration (events per minute)
            events_per_minute = (count / duration) * 60
            total_weighted_score += weight * min(events_per_minute, 5.0)  # Cap at 5 events/min
        
        # Scale to 0-10 range
        return min(total_weighted_score, 10.0)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to level"""
        if risk_score >= 8.0:
            return "critical"
        elif risk_score >= 6.0:
            return "high"
        elif risk_score >= 4.0:
            return "medium"
        elif risk_score >= 2.0:
            return "low"
        else:
            return "minimal"
    
    def _generate_recommendations(self, summary: Dict) -> List[str]:
        """Generate recommendations based on detected events"""
        recommendations = []
        
        if summary.get('Phone Usage Detected', 0) > 0:
            recommendations.append("Phone usage detected - implement stricter device policies")
            
        if summary.get('head_turn_frequent', 0) > 0:
            recommendations.append("Frequent head turning observed - review seating arrangements")
            
        if summary.get('head_turn_sustained', 0) > 0:
            recommendations.append("Sustained head turning detected - monitor for collaboration")
            
        if summary.get('hand_extended_duration', 0) > 0:
            recommendations.append("Extended hand gestures observed - check for unauthorized materials")
        
        if not recommendations:
            recommendations.append("No significant suspicious activity detected")
        
        return recommendations
