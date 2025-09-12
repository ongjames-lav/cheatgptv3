/**
 * CheatGPT Playback JavaScript
 * Handles video playback with interactive hotspot markers
 */

class CheatGPTPlayback {
    constructor(videoElement, timelineElement, eventsData) {
        this.video = videoElement;
        this.timeline = timelineElement;
        this.events = eventsData || [];
        this.markers = [];
        this.isPlaying = false;
        this.isDragging = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.createMarkers();
        this.setupKeyboardShortcuts();
        this.updateTimeDisplay();
    }
    
    setupEventListeners() {
        // Video events
        this.video.addEventListener('loadedmetadata', () => {
            this.createMarkers();
            this.updateTimeDisplay();
        });
        
        this.video.addEventListener('timeupdate', () => {
            this.updateProgress();
            this.checkForActiveEvents();
        });
        
        this.video.addEventListener('play', () => {
            this.isPlaying = true;
            this.updatePlayButton();
        });
        
        this.video.addEventListener('pause', () => {
            this.isPlaying = false;
            this.updatePlayButton();
        });
        
        this.video.addEventListener('ended', () => {
            this.isPlaying = false;
            this.updatePlayButton();
        });
        
        // Timeline events
        if (this.timeline) {
            this.timeline.addEventListener('click', (e) => {
                if (!this.isDragging) {
                    this.seekToPosition(e);
                }
            });
            
            this.timeline.addEventListener('mousedown', () => {
                this.isDragging = true;
            });
            
            document.addEventListener('mouseup', () => {
                this.isDragging = false;
            });
        }
        
        // Control buttons
        this.setupControlButtons();
    }
    
    setupControlButtons() {
        const playBtn = document.getElementById('play-btn');
        const pauseBtn = document.getElementById('pause-btn');
        const skipBackBtn = document.getElementById('skip-back-btn');
        const skipForwardBtn = document.getElementById('skip-forward-btn');
        const fullscreenBtn = document.getElementById('fullscreen-btn');
        
        if (playBtn) {
            playBtn.addEventListener('click', () => this.play());
        }
        
        if (pauseBtn) {
            pauseBtn.addEventListener('click', () => this.pause());
        }
        
        if (skipBackBtn) {
            skipBackBtn.addEventListener('click', () => this.skipBackward());
        }
        
        if (skipForwardBtn) {
            skipForwardBtn.addEventListener('click', () => this.skipForward());
        }
        
        if (fullscreenBtn) {
            fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());
        }
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Only handle shortcuts when not typing in an input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }
            
            switch (e.code) {
                case 'Space':
                    e.preventDefault();
                    this.togglePlayPause();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    this.skipBackward();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.skipForward();
                    break;
                case 'KeyF':
                    e.preventDefault();
                    this.toggleFullscreen();
                    break;
                case 'KeyM':
                    e.preventDefault();
                    this.toggleMute();
                    break;
                case 'Home':
                    e.preventDefault();
                    this.seekTo(0);
                    break;
                case 'End':
                    e.preventDefault();
                    this.seekTo(this.video.duration);
                    break;
            }
        });
    }
    
    createMarkers() {
        if (!this.timeline || !this.video.duration) {
            return;
        }
        
        // Clear existing markers
        const existingMarkers = this.timeline.querySelectorAll('.timeline-marker');
        existingMarkers.forEach(marker => marker.remove());
        this.markers = [];
        
        // Create new markers
        this.events.forEach((event, index) => {
            const marker = this.createMarker(event, index);
            if (marker) {
                this.timeline.appendChild(marker);
                this.markers.push(marker);
            }
        });
    }
    
    createMarker(event, index) {
        const marker = document.createElement('div');
        marker.className = `timeline-marker marker-${event.event_type.replace('_', '-')}`;
        
        // Calculate position based on timestamp
        const position = (event.timestamp_offset / this.video.duration) * 100;
        marker.style.left = `${position}%`;
        
        // Add tooltip
        const confidence = Math.round(event.confidence * 100);
        const timeStr = this.formatTime(event.timestamp_offset);
        marker.title = `${event.event_type.replace('_', ' ')} (${confidence}%) at ${timeStr}`;
        
        // Add event listeners
        marker.addEventListener('click', (e) => {
            e.stopPropagation();
            this.seekToEvent(event);
            this.highlightEvent(event, index);
        });
        
        marker.addEventListener('mouseenter', () => {
            this.showEventTooltip(event, marker);
        });
        
        marker.addEventListener('mouseleave', () => {
            this.hideEventTooltip();
        });
        
        return marker;
    }
    
    seekToPosition(e) {
        if (!this.video.duration) return;
        
        const rect = this.timeline.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const percentage = clickX / rect.width;
        const time = percentage * this.video.duration;
        
        this.seekTo(time);
    }
    
    seekTo(time) {
        if (this.video.duration) {
            this.video.currentTime = Math.max(0, Math.min(time, this.video.duration));
        }
    }
    
    seekToEvent(event) {
        this.seekTo(event.timestamp_offset);
        
        // Auto-play if paused
        if (this.video.paused) {
            this.play();
        }
    }
    
    play() {
        const playPromise = this.video.play();
        if (playPromise !== undefined) {
            playPromise.catch(error => {
                console.warn('Play failed:', error);
            });
        }
    }
    
    pause() {
        this.video.pause();
    }
    
    togglePlayPause() {
        if (this.video.paused) {
            this.play();
        } else {
            this.pause();
        }
    }
    
    skipBackward(seconds = 10) {
        this.seekTo(this.video.currentTime - seconds);
    }
    
    skipForward(seconds = 10) {
        this.seekTo(this.video.currentTime + seconds);
    }
    
    toggleFullscreen() {
        if (!document.fullscreenElement) {
            if (this.video.requestFullscreen) {
                this.video.requestFullscreen();
            } else if (this.video.webkitRequestFullscreen) {
                this.video.webkitRequestFullscreen();
            } else if (this.video.msRequestFullscreen) {
                this.video.msRequestFullscreen();
            }
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
                document.webkitExitFullscreen();
            } else if (document.msExitFullscreen) {
                document.msExitFullscreen();
            }
        }
    }
    
    toggleMute() {
        this.video.muted = !this.video.muted;
        this.updateMuteButton();
    }
    
    updateProgress() {
        if (!this.video.duration || this.isDragging) return;
        
        const progress = (this.video.currentTime / this.video.duration) * 100;
        const progressBar = document.querySelector('.timeline-progress');
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
        
        this.updateTimeDisplay();
    }
    
    updateTimeDisplay() {
        const currentTimeEl = document.getElementById('current-time');
        const durationEl = document.getElementById('duration');
        
        if (currentTimeEl) {
            currentTimeEl.textContent = this.formatTime(this.video.currentTime || 0);
        }
        
        if (durationEl) {
            durationEl.textContent = this.formatTime(this.video.duration || 0);
        }
    }
    
    updatePlayButton() {
        const playBtn = document.getElementById('play-btn');
        const pauseBtn = document.getElementById('pause-btn');
        
        if (this.isPlaying) {
            if (playBtn) playBtn.style.display = 'none';
            if (pauseBtn) pauseBtn.style.display = 'inline-flex';
        } else {
            if (playBtn) playBtn.style.display = 'inline-flex';
            if (pauseBtn) pauseBtn.style.display = 'none';
        }
    }
    
    updateMuteButton() {
        const muteBtn = document.getElementById('mute-btn');
        const unmuteBtn = document.getElementById('unmute-btn');
        
        if (this.video.muted) {
            if (muteBtn) muteBtn.style.display = 'none';
            if (unmuteBtn) unmuteBtn.style.display = 'inline-flex';
        } else {
            if (muteBtn) muteBtn.style.display = 'inline-flex';
            if (unmuteBtn) unmuteBtn.style.display = 'none';
        }
    }
    
    checkForActiveEvents() {
        const currentTime = this.video.currentTime;
        const tolerance = 2; // 2 seconds tolerance
        
        // Find events near current time
        const activeEvents = this.events.filter(event => {
            return Math.abs(currentTime - event.timestamp_offset) <= tolerance;
        });
        
        // Update UI for active events
        this.updateActiveEventDisplay(activeEvents);
    }
    
    updateActiveEventDisplay(activeEvents) {
        const eventDisplay = document.getElementById('current-event');
        if (!eventDisplay) return;
        
        if (activeEvents.length > 0) {
            const event = activeEvents[0]; // Show most recent event
            const confidence = Math.round(event.confidence * 100);
            eventDisplay.innerHTML = `
                <div class="active-event">
                    <strong>${event.event_type.replace('_', ' ').toUpperCase()}</strong>
                    <span class="confidence">${confidence}% confidence</span>
                </div>
            `;
            eventDisplay.style.display = 'block';
        } else {
            eventDisplay.style.display = 'none';
        }
    }
    
    highlightEvent(event, index) {
        // Highlight in events list
        const eventItems = document.querySelectorAll('.event-item');
        eventItems.forEach(item => item.classList.remove('active'));
        
        if (eventItems[index]) {
            eventItems[index].classList.add('active');
            eventItems[index].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }
    
    showEventTooltip(event, marker) {
        // Create or update tooltip
        let tooltip = document.getElementById('event-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'event-tooltip';
            tooltip.className = 'event-tooltip';
            document.body.appendChild(tooltip);
        }
        
        const confidence = Math.round(event.confidence * 100);
        const timeStr = this.formatTime(event.timestamp_offset);
        
        tooltip.innerHTML = `
            <div class="tooltip-header">${event.event_type.replace('_', ' ').toUpperCase()}</div>
            <div class="tooltip-content">
                <div>Time: ${timeStr}</div>
                <div>Confidence: ${confidence}%</div>
            </div>
        `;
        
        // Position tooltip near marker
        const markerRect = marker.getBoundingClientRect();
        tooltip.style.left = `${markerRect.left + markerRect.width / 2}px`;
        tooltip.style.top = `${markerRect.top - tooltip.offsetHeight - 10}px`;
        tooltip.style.display = 'block';
    }
    
    hideEventTooltip() {
        const tooltip = document.getElementById('event-tooltip');
        if (tooltip) {
            tooltip.style.display = 'none';
        }
    }
    
    formatTime(seconds) {
        if (isNaN(seconds)) return '00:00';
        
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    
    // Public API methods
    getCurrentTime() {
        return this.video.currentTime;
    }
    
    getDuration() {
        return this.video.duration;
    }
    
    getEvents() {
        return this.events;
    }
    
    addEvent(event) {
        this.events.push(event);
        this.createMarkers(); // Recreate markers
    }
    
    removeEvent(index) {
        if (index >= 0 && index < this.events.length) {
            this.events.splice(index, 1);
            this.createMarkers(); // Recreate markers
        }
    }
    
    destroy() {
        // Clean up event listeners and DOM elements
        const tooltip = document.getElementById('event-tooltip');
        if (tooltip) {
            tooltip.remove();
        }
        
        this.markers.forEach(marker => marker.remove());
        this.markers = [];
    }
}

// Utility functions for event management
class EventManager {
    constructor() {
        this.eventListeners = [];
    }
    
    static formatEventType(eventType) {
        return eventType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    
    static getEventColor(eventType) {
        const colors = {
            'suspicious_gesture': '#ff00ff',
            'suspicious_looking': '#ffa500',
            'suspicious_lean': '#ffff00',
            'phone_detected': '#ff0000',
            'default': '#007acc'
        };
        return colors[eventType] || colors.default;
    }
    
    static getConfidenceClass(confidence) {
        if (confidence > 0.8) return 'confidence-high';
        if (confidence > 0.6) return 'confidence-medium';
        return 'confidence-low';
    }
    
    static createEventListItem(event, index, onClickCallback) {
        const item = document.createElement('div');
        item.className = 'event-item';
        item.dataset.index = index;
        
        const confidence = Math.round(event.confidence * 100);
        const timeStr = CheatGPTPlayback.prototype.formatTime(event.timestamp_offset);
        
        item.innerHTML = `
            <div class="event-details">
                <h4>${this.formatEventType(event.event_type)}</h4>
                <div class="event-meta">
                    Time: ${timeStr} | Frame: ${event.frame_no || 'N/A'}
                </div>
            </div>
            <div class="event-confidence ${this.getConfidenceClass(event.confidence)}">
                ${confidence}%
            </div>
        `;
        
        if (onClickCallback) {
            item.addEventListener('click', () => onClickCallback(event, index));
        }
        
        return item;
    }
}

// Initialize playback when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('main-video');
    const timeline = document.querySelector('.timeline');
    
    // Get events data from page (should be set by template)
    let eventsData = [];
    if (typeof window.sessionEvents !== 'undefined') {
        eventsData = window.sessionEvents;
    }
    
    if (video) {
        window.playbackInstance = new CheatGPTPlayback(video, timeline, eventsData);
        
        // Populate events list if container exists
        const eventsList = document.getElementById('events-list');
        if (eventsList && eventsData.length > 0) {
            eventsList.innerHTML = '';
            eventsData.forEach((event, index) => {
                const eventItem = EventManager.createEventListItem(
                    event, 
                    index, 
                    (event, index) => {
                        window.playbackInstance.seekToEvent(event);
                        window.playbackInstance.highlightEvent(event, index);
                    }
                );
                eventsList.appendChild(eventItem);
            });
        }
    }
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CheatGPTPlayback, EventManager };
}
