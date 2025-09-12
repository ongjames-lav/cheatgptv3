// Test the new realistic suspicious activity calculation
function testRealisticCalculation() {
    // Test sessions based on real data we found
    const testCases = [
        { events: 11, duration: 38.6, critical: 2, description: "Short test session with device detection" },
        { events: 7, duration: 21.3, critical: 3, description: "Short session with high critical events" },
        { events: 10, duration: 34.2, critical: 5, description: "Short session with many device detections" },
        { events: 3, duration: 1800, critical: 0, description: "30-min normal exam session" },
        { events: 8, duration: 3600, critical: 1, description: "60-min exam with some suspicious activity" },
        { events: 15, duration: 2400, critical: 2, description: "40-min exam with multiple issues" },
        { events: 25, duration: 1800, critical: 5, description: "30-min highly suspicious session" },
        { events: 0, duration: 1800, critical: 0, description: "Perfect 30-min session" },
    ];

    console.log("Realistic Suspicious Activity Calculation Test:");
    console.log("===============================================");

    testCases.forEach(test => {
        const eventsPerMinute = test.events / (test.duration / 60);
        const sessionMinutes = test.duration / 60;
        
        let adjustedEventsPerMin = eventsPerMinute;
        
        // For very short test sessions, normalize the rate
        if (sessionMinutes < 2) {
            adjustedEventsPerMin = eventsPerMinute * 0.6;
        }
        
        let suspiciousPercentage = 0;
        
        if (adjustedEventsPerMin <= 0.1) {
            suspiciousPercentage = Math.round(adjustedEventsPerMin * 250); // 0-25%
        } else if (adjustedEventsPerMin <= 0.27) {
            suspiciousPercentage = Math.round(25 + (adjustedEventsPerMin - 0.1) * 147); // 25-50%
        } else if (adjustedEventsPerMin <= 0.67) {
            suspiciousPercentage = Math.round(50 + (adjustedEventsPerMin - 0.27) * 62.5); // 50-75%
        } else {
            suspiciousPercentage = Math.round(75 + Math.min(25, (adjustedEventsPerMin - 0.67) * 75)); // 75-100%
        }
        
        // Factor in critical events
        if (test.critical > 0) {
            const criticalBonus = Math.min(30, test.critical * 15);
            suspiciousPercentage = Math.min(100, suspiciousPercentage + criticalBonus);
        }
        
        suspiciousPercentage = Math.max(0, Math.min(100, suspiciousPercentage));

        console.log(`${test.description}:`);
        console.log(`  ${test.events} events in ${sessionMinutes.toFixed(1)} min = ${eventsPerMinute.toFixed(2)}/min (adj: ${adjustedEventsPerMin.toFixed(2)}/min)`);
        console.log(`  Critical events: ${test.critical}, Final result: ${suspiciousPercentage}%`);
        console.log("");
    });
}

// Run the test
testRealisticCalculation();
