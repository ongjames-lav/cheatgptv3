// Test the new suspicious activity calculation
function testSuspiciousCalculation() {
    const testCases = [
        { events: 0, duration: 300, description: "No events in 5 mins" },
        { events: 1, duration: 300, description: "1 event in 5 mins (0.2/min)" },
        { events: 2, duration: 300, description: "2 events in 5 mins (0.4/min)" },
        { events: 3, duration: 300, description: "3 events in 5 mins (0.6/min)" },
        { events: 5, duration: 300, description: "5 events in 5 mins (1/min)" },
        { events: 10, duration: 300, description: "10 events in 5 mins (2/min)" },
        { events: 15, duration: 300, description: "15 events in 5 mins (3/min)" },
        { events: 2, duration: 60, description: "2 events in 1 min (2/min)" },
        { events: 30, duration: 30, description: "30 events in 30 sec (60/min)" },
    ];

    console.log("Suspicious Activity Calculation Test:");
    console.log("=====================================");

    testCases.forEach(test => {
        const eventsPerMinute = test.events / (test.duration / 60);
        let suspiciousPercentage = 0;

        if (eventsPerMinute <= 0.5) {
            suspiciousPercentage = Math.round(eventsPerMinute * 50);
        } else if (eventsPerMinute <= 1.0) {
            suspiciousPercentage = Math.round(25 + (eventsPerMinute - 0.5) * 50);
        } else if (eventsPerMinute <= 2.0) {
            suspiciousPercentage = Math.round(50 + (eventsPerMinute - 1.0) * 25);
        } else {
            suspiciousPercentage = Math.round(75 + Math.min(25, (eventsPerMinute - 2.0) * 12.5));
        }

        suspiciousPercentage = Math.max(0, Math.min(100, suspiciousPercentage));

        console.log(`${test.description}: ${eventsPerMinute.toFixed(1)}/min = ${suspiciousPercentage}%`);
    });
}

// Run the test
testSuspiciousCalculation();
