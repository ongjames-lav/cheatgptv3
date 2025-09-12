// Test the new behavior scoring system
function testBehaviorScoring() {
    const testCases = [
        { events: 0, duration: 1800, critical: 0, warning: 0, description: "Perfect 30-min exam" },
        { events: 2, duration: 3600, critical: 0, warning: 1, description: "Normal 60-min exam with minor movement" },
        { events: 5, duration: 1800, critical: 0, warning: 3, description: "Slightly fidgety 30-min session" },
        { events: 8, duration: 2400, critical: 1, warning: 4, description: "40-min exam with device detection" },
        { events: 15, duration: 1800, critical: 2, warning: 8, description: "Concerning 30-min session" },
        { events: 25, duration: 1200, critical: 5, warning: 15, description: "Highly suspicious 20-min session" },
        { events: 11, duration: 38.6, critical: 2, warning: 9, description: "Real test session from our data" },
    ];

    console.log("Behavior Scoring System Test:");
    console.log("============================");

    testCases.forEach(test => {
        const eventsPerMinute = test.events / (test.duration / 60);
        const sessionMinutes = test.duration / 60;
        
        let behaviorScore = { level: 'Normal', color: 'green', icon: '✓', description: 'No issues detected' };
        
        // Apply the same logic as in our HTML
        if (test.critical >= 3 || (test.critical >= 1 && eventsPerMinute > 5)) {
            behaviorScore = { 
                level: 'Critical', 
                color: 'red', 
                icon: '🚨', 
                description: 'Serious violations detected' 
            };
        } else if (test.critical >= 1 || test.warning >= 5 || eventsPerMinute > 3) {
            behaviorScore = { 
                level: 'Flagged', 
                color: 'orange', 
                icon: '⚠️', 
                description: 'Suspicious activity detected' 
            };
        } else if (test.warning >= 2 || eventsPerMinute > 1) {
            behaviorScore = { 
                level: 'Review', 
                color: 'yellow', 
                icon: '👁️', 
                description: 'Minor concerns noted' 
            };
        } else if (test.events > 0) {
            behaviorScore = { 
                level: 'Normal+', 
                color: 'blue', 
                icon: '📋', 
                description: 'Typical exam behavior' 
            };
        }

        console.log(`${test.description}:`);
        console.log(`  ${test.events} events in ${sessionMinutes.toFixed(1)} min (${eventsPerMinute.toFixed(1)}/min)`);
        console.log(`  Critical: ${test.critical}, Warning: ${test.warning}`);
        console.log(`  Result: ${behaviorScore.icon} ${behaviorScore.level} (${behaviorScore.color}) - ${behaviorScore.description}`);
        console.log("");
    });
}

// Run the test
testBehaviorScoring();
