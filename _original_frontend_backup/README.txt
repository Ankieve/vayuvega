PRISM-TC Frontend
==================

Folder structure:
PRISM-TC-Frontend/
├── index.html
├── style.css
└── app.js

The three files are linked as:
index.html -> style.css
index.html -> app.js

External libraries are loaded from CDNs:
- Leaflet 1.9.4
- Chart.js
- Google Fonts (Inter)

IMPORTANT:
The frontend JavaScript is configured to call:
  /api/health
  /api/demo
  /api/predict

So the frontend needs a backend/proxy serving those API routes for the AI
prediction buttons to work.

To test only the UI:
1. Open this folder in VS Code.
2. Install the VS Code "Live Server" extension.
3. Right-click index.html.
4. Choose "Open with Live Server".

The page itself should load, but AI status/demo/prediction will require
the backend API.

For a full project, keep this frontend folder alongside your backend,
or configure your backend/proxy so /api/* routes reach the prediction server.
