"""
Session Report Generator
Creates PDF reports for CheatGPT detection sessions
"""

import os
import time
import tempfile
from datetime import datetime
from typing import Dict, List, Optional
import logging

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.platypus.flowables import PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

class SessionReportGenerator:
    """Generate PDF reports for detection sessions"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.styles = None
        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()
            self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        if not self.styles:
            return
            
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2c3e50'),
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#34495e'),
            alignment=TA_LEFT
        ))
        
        # Alert style for high-risk events
        self.styles.add(ParagraphStyle(
            name='AlertText',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.red,
            backColor=colors.HexColor('#ffebee'),
            borderColor=colors.red,
            borderWidth=1,
            borderPadding=5
        ))
    
    def generate_report(self, session: Dict, hotspots: List[Dict]) -> Optional[str]:
        """Generate a comprehensive PDF report for a session"""
        if not REPORTLAB_AVAILABLE:
            logger.error("ReportLab not available - cannot generate PDF reports")
            return None
        
        try:
            # Create PDF file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_filename = f"cheatgpt_report_{session['session_id']}_{timestamp}.pdf"
            pdf_path = os.path.join(self.temp_dir, pdf_filename)
            
            doc = SimpleDocTemplate(pdf_path, pagesize=A4, 
                                  topMargin=1*inch, bottomMargin=1*inch,
                                  leftMargin=0.75*inch, rightMargin=0.75*inch)
            
            story = []
            
            # Title Page
            story.extend(self._create_title_page(session, hotspots))
            
            # Executive Summary
            story.extend(self._create_executive_summary(session, hotspots))
            
            # Session Details
            story.extend(self._create_session_details(session))
            
            # Hotspot Analysis
            story.extend(self._create_hotspot_analysis(hotspots))
            
            # Timeline
            story.extend(self._create_timeline_section(session, hotspots))
            
            # Risk Assessment
            story.extend(self._create_risk_assessment(hotspots))
            
            # Recommendations
            story.extend(self._create_recommendations(hotspots))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Generated report: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None
    
    def _create_title_page(self, session: Dict, hotspots: List[Dict]) -> List:
        """Create the title page"""
        story = []
        
        # Main title
        story.append(Paragraph("CheatGPT Detection Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Session info table
        session_data = [
            ['Session ID:', session['session_id']],
            ['Date:', datetime.fromtimestamp(session['start_ts']).strftime('%Y-%m-%d %H:%M:%S')],
            ['Duration:', f"{session.get('duration', 0):.1f} seconds"],
            ['Total Events:', str(len(hotspots))],
            ['Status:', session.get('status', 'Unknown').title()]
        ]
        
        session_table = Table(session_data, colWidths=[2*inch, 3*inch])
        session_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        
        story.append(session_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Risk level indicator
        risk_level = self._calculate_risk_level(hotspots)
        risk_color = self._get_risk_color(risk_level)
        
        story.append(Paragraph(f"<b>Risk Level: <font color='{risk_color}'>{risk_level.upper()}</font></b>", 
                              self.styles['Heading2']))
        
        story.append(PageBreak())
        return story
    
    def _create_executive_summary(self, session: Dict, hotspots: List[Dict]) -> List:
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['CustomSubtitle']))
        
        # Calculate statistics
        total_events = len(hotspots)
        high_confidence_events = len([h for h in hotspots if h['confidence'] > 0.8])
        event_types = {}
        
        for hotspot in hotspots:
            event_type = hotspot['event_type']
            if event_type not in event_types:
                event_types[event_type] = 0
            event_types[event_type] += 1
        
        # Summary text
        summary_text = f"""
        This report analyzes a {session.get('duration', 0):.1f}-second monitoring session 
        that detected {total_events} suspicious events. Of these, {high_confidence_events} 
        events were classified as high-confidence detections (>80% confidence).
        """
        
        if event_types:
            most_common = max(event_types.items(), key=lambda x: x[1])
            summary_text += f"""
            <br/><br/>
            The most frequently detected behavior was <b>{most_common[0].replace('_', ' ')}</b> 
            with {most_common[1]} occurrences.
            """
        
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        return story
    
    def _create_session_details(self, session: Dict) -> List:
        """Create session details section"""
        story = []
        
        story.append(Paragraph("Session Details", self.styles['CustomSubtitle']))
        
        # Detailed session info
        start_time = datetime.fromtimestamp(session['start_ts'])
        end_time = datetime.fromtimestamp(session['end_ts']) if session.get('end_ts') else None
        
        details_data = [
            ['Start Time', start_time.strftime('%Y-%m-%d %H:%M:%S')],
            ['End Time', end_time.strftime('%Y-%m-%d %H:%M:%S') if end_time else 'N/A'],
            ['Duration', f"{session.get('duration', 0):.2f} seconds"],
            ['Video File', os.path.basename(session.get('video_path', 'N/A'))],
            ['Recording Status', session.get('status', 'Unknown').title()]
        ]
        
        if session.get('metadata'):
            metadata = session['metadata']
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    details_data.append([key.replace('_', ' ').title(), str(value)])
        
        details_table = Table(details_data, colWidths=[2*inch, 4*inch])
        details_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        
        story.append(details_table)
        story.append(Spacer(1, 0.3*inch))
        
        return story
    
    def _create_hotspot_analysis(self, hotspots: List[Dict]) -> List:
        """Create hotspot analysis section"""
        story = []
        
        story.append(Paragraph("Event Analysis", self.styles['CustomSubtitle']))
        
        if not hotspots:
            story.append(Paragraph("No suspicious events detected during this session.", 
                                 self.styles['Normal']))
            return story
        
        # Event summary table
        event_types = {}
        confidence_ranges = {'High (>80%)': 0, 'Medium (60-80%)': 0, 'Low (<60%)': 0}
        
        for hotspot in hotspots:
            event_type = hotspot['event_type'].replace('_', ' ').title()
            confidence = hotspot['confidence']
            
            if event_type not in event_types:
                event_types[event_type] = {'count': 0, 'avg_confidence': 0, 'confidences': []}
            
            event_types[event_type]['count'] += 1
            event_types[event_type]['confidences'].append(confidence)
            
            # Categorize by confidence
            if confidence > 0.8:
                confidence_ranges['High (>80%)'] += 1
            elif confidence > 0.6:
                confidence_ranges['Medium (60-80%)'] += 1
            else:
                confidence_ranges['Low (<60%)'] += 1
        
        # Calculate average confidences
        for event_type in event_types:
            confidences = event_types[event_type]['confidences']
            event_types[event_type]['avg_confidence'] = sum(confidences) / len(confidences)
        
        # Create summary table
        summary_data = [['Event Type', 'Count', 'Avg Confidence']]
        for event_type, data in sorted(event_types.items(), key=lambda x: x[1]['count'], reverse=True):
            summary_data.append([
                event_type,
                str(data['count']),
                f"{data['avg_confidence']:.1%}"
            ])
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 1*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Confidence distribution
        story.append(Paragraph("Confidence Distribution:", self.styles['Heading3']))
        conf_data = [['Confidence Level', 'Count', 'Percentage']]
        total_events = len(hotspots)
        
        for level, count in confidence_ranges.items():
            percentage = (count / total_events * 100) if total_events > 0 else 0
            conf_data.append([level, str(count), f"{percentage:.1f}%"])
        
        conf_table = Table(conf_data, colWidths=[2*inch, 1*inch, 1*inch])
        conf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        
        story.append(conf_table)
        story.append(Spacer(1, 0.3*inch))
        
        return story
    
    def _create_timeline_section(self, session: Dict, hotspots: List[Dict]) -> List:
        """Create timeline section with events"""
        story = []
        
        story.append(Paragraph("Event Timeline", self.styles['CustomSubtitle']))
        
        if not hotspots:
            story.append(Paragraph("No events to display in timeline.", self.styles['Normal']))
            return story
        
        # Sort hotspots by timestamp
        sorted_hotspots = sorted(hotspots, key=lambda x: x['timestamp_offset'])
        
        # Create timeline table
        timeline_data = [['Time (s)', 'Event Type', 'Confidence', 'Frame']]
        
        for hotspot in sorted_hotspots:
            timeline_data.append([
                f"{hotspot['timestamp_offset']:.1f}",
                hotspot['event_type'].replace('_', ' ').title(),
                f"{hotspot['confidence']:.1%}",
                str(hotspot.get('frame_no', 'N/A'))
            ])
        
        timeline_table = Table(timeline_data, colWidths=[1*inch, 2*inch, 1*inch, 1*inch])
        timeline_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
            # Highlight high-confidence events
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fff3cd')),  # Default
        ]))
        
        # Apply conditional formatting for confidence levels
        for i, hotspot in enumerate(sorted_hotspots, 1):
            if hotspot['confidence'] > 0.8:
                timeline_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f8d7da'))  # High confidence - red
                ]))
            elif hotspot['confidence'] > 0.6:
                timeline_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, i), (-1, i), colors.HexColor('#fff3cd'))  # Medium confidence - yellow
                ]))
        
        story.append(timeline_table)
        story.append(Spacer(1, 0.3*inch))
        
        return story
    
    def _create_risk_assessment(self, hotspots: List[Dict]) -> List:
        """Create risk assessment section"""
        story = []
        
        story.append(Paragraph("Risk Assessment", self.styles['CustomSubtitle']))
        
        risk_level = self._calculate_risk_level(hotspots)
        risk_color = self._get_risk_color(risk_level)
        
        # Risk level
        story.append(Paragraph(f"<b>Overall Risk Level: <font color='{risk_color}'>{risk_level.upper()}</font></b>", 
                              self.styles['Heading3']))
        
        # Risk factors
        risk_factors = []
        
        high_conf_events = len([h for h in hotspots if h['confidence'] > 0.8])
        if high_conf_events > 0:
            risk_factors.append(f"• {high_conf_events} high-confidence events detected")
        
        phone_events = len([h for h in hotspots if 'phone' in h['event_type'].lower()])
        if phone_events > 0:
            risk_factors.append(f"• {phone_events} phone usage events detected")
        
        gesture_events = len([h for h in hotspots if 'gesture' in h['event_type'].lower()])
        if gesture_events > 0:
            risk_factors.append(f"• {gesture_events} suspicious gesture events detected")
        
        if len(hotspots) > 10:
            risk_factors.append(f"• High frequency of events ({len(hotspots)} total)")
        
        if risk_factors:
            risk_text = "<br/>".join(risk_factors)
            story.append(Paragraph(f"<b>Risk Factors:</b><br/>{risk_text}", self.styles['Normal']))
        else:
            story.append(Paragraph("No significant risk factors identified.", self.styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        return story
    
    def _create_recommendations(self, hotspots: List[Dict]) -> List:
        """Create recommendations section"""
        story = []
        
        story.append(Paragraph("Recommendations", self.styles['CustomSubtitle']))
        
        recommendations = []
        
        # Analyze patterns and generate recommendations
        if len(hotspots) == 0:
            recommendations.append("• No suspicious activity detected - continue monitoring")
        else:
            high_conf_events = len([h for h in hotspots if h['confidence'] > 0.8])
            
            if high_conf_events > 3:
                recommendations.append("• Immediate review recommended due to multiple high-confidence events")
            elif high_conf_events > 0:
                recommendations.append("• Review high-confidence events for verification")
            
            phone_events = len([h for h in hotspots if 'phone' in h['event_type'].lower()])
            if phone_events > 0:
                recommendations.append("• Investigate phone usage incidents")
                recommendations.append("• Consider stricter phone policies during monitoring")
            
            gesture_events = len([h for h in hotspots if 'gesture' in h['event_type'].lower()])
            if gesture_events > 2:
                recommendations.append("• Review camera angles and positioning")
                recommendations.append("• Consider additional monitoring for suspicious movements")
            
            if len(hotspots) > 10:
                recommendations.append("• High event frequency - consider extending monitoring duration")
                recommendations.append("• Review detection sensitivity settings")
        
        # General recommendations
        recommendations.extend([
            "• Maintain recording equipment in optimal condition",
            "• Ensure adequate lighting for detection accuracy",
            "• Regular calibration of detection systems recommended"
        ])
        
        recommendations_text = "<br/>".join(recommendations)
        story.append(Paragraph(recommendations_text, self.styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        story.append(Paragraph(f"<i>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by CheatGPT Detection System</i>", 
                              self.styles['Normal']))
        
        return story
    
    def _calculate_risk_level(self, hotspots: List[Dict]) -> str:
        """Calculate overall risk level based on events"""
        if not hotspots:
            return "low"
        
        high_conf_events = len([h for h in hotspots if h['confidence'] > 0.8])
        total_events = len(hotspots)
        
        # Phone usage is high risk
        phone_events = len([h for h in hotspots if 'phone' in h['event_type'].lower()])
        
        if phone_events > 0 or high_conf_events > 3:
            return "high"
        elif high_conf_events > 1 or total_events > 5:
            return "medium"
        else:
            return "low"
    
    def _get_risk_color(self, risk_level: str) -> str:
        """Get color for risk level"""
        colors_map = {
            'low': '#27ae60',    # Green
            'medium': '#f39c12', # Orange
            'high': '#e74c3c'    # Red
        }
        return colors_map.get(risk_level, '#7f8c8d')  # Default gray

if __name__ == "__main__":
    # Test report generation
    print("Testing SessionReportGenerator...")
    
    if not REPORTLAB_AVAILABLE:
        print("❌ ReportLab not available")
    else:
        print("✅ ReportLab available")
    
    # Mock data for testing
    mock_session = {
        'session_id': 'test_session_001',
        'start_ts': time.time() - 300,  # 5 minutes ago
        'end_ts': time.time(),
        'duration': 300,
        'status': 'completed',
        'video_path': '/videos/test_session_001.mp4',
        'metadata': {'test': True}
    }
    
    mock_hotspots = [
        {
            'event_type': 'suspicious_gesture',
            'confidence': 0.85,
            'timestamp_offset': 45.2,
            'frame_no': 1356
        },
        {
            'event_type': 'phone_detected',
            'confidence': 0.92,
            'timestamp_offset': 120.8,
            'frame_no': 3624
        }
    ]
    
    generator = SessionReportGenerator()
    if REPORTLAB_AVAILABLE:
        pdf_path = generator.generate_report(mock_session, mock_hotspots)
        if pdf_path:
            print(f"✅ Test report generated: {pdf_path}")
        else:
            print("❌ Failed to generate test report")
    else:
        print("Skipping test - ReportLab not available")
