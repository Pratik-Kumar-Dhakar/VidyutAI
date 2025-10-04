
// A simple WebSocket server to stream logs.
// To run this, you need to have Node.js installed.
// 1. Save this file as server.js
// 2. Open a terminal in the same directory.
// 3. Run `npm install ws` to install the required library.
// 4. Run `node server.js` to start the server.

const WebSocket = require('ws');

// Initialize a WebSocket server on port 8080
const wss = new WebSocket.Server({ port: 8080 });

console.log('WebSocket log server started on ws://localhost:8080');

// Mock data for generating random logs
const logLevels = ['INFO', 'WARN', 'ERROR', 'SUCCESS'];
const logMessages = [
    'Initializing system services...',
    'Database connection successful.',
    'User `admin` logged in from IP 192.168.1.1',
    'API endpoint GET /api/users reached.',
    'Data sync initiated with primary server.',
    'High memory usage detected on worker-03.',
    'Scheduled backup completed.',
    'Failed to connect to payment gateway.',
    'dil me mere hai darde disco',
    'darde disco, darde disco',
    'apt apt apt'
];

// This function runs whenever a new client connects to the server.
wss.on('connection', ws => {
    console.log('Dashboard client connected');

    // Send a welcome message to the newly connected client.
    ws.send(JSON.stringify({
        level: 'INFO',
        message: 'Successfully connected to the live log stream.'
    }));

    // Set up an interval to periodically send a new log message.
    const intervalId = setInterval(() => {
        // Create a random log entry.
        const level = logLevels[Math.floor(Math.random() * logLevels.length)];
        const message = logMessages[Math.floor(Math.random() * logMessages.length)];
        const logData = { level, message };

        // Send the log data to the client as a JSON string.
        ws.send(JSON.stringify(logData));
    }, Math.random() * (4000 - 1500) + 1500); // Send logs at a random interval between 1.5 and 4 seconds.

    // This function runs when a client disconnects.
    ws.on('close', () => {
        console.log('Dashboard client disconnected');
        // Stop sending messages to prevent memory leaks.
        clearInterval(intervalId);
    });

    // Handle any errors that occur.
    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
        clearInterval(intervalId);
    });
});
