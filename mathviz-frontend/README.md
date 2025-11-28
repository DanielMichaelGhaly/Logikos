# 🧮 MathViz Modern Frontend

A beautiful, modern React/TypeScript frontend for MathViz with:

- ✨ **Beautiful UI** - Modern design with smooth animations
- 🧮 **LaTeX Math Rendering** - Professional mathematical notation using KaTeX
- 📊 **Interactive Graphs** - Embedded Desmos calculator integration
- 📱 **Responsive Design** - Works perfectly on desktop, tablet, and mobile
- 🎯 **Step-by-Step Solutions** - Clear, expandable solution breakdowns
- ⚡ **Fast Performance** - Optimized React components with Framer Motion

## 🚀 Quick Start

### Prerequisites

- Node.js 16+ and npm
- Python 3.8+ (for the backend)

### 1. Install Dependencies

```bash
cd mathviz-frontend
npm install
```

### 2. Start the Backend Server

In a separate terminal:

```bash
cd ../mathviz-backend
pip install -r requirements.txt
python main.py
```

The backend will start at `http://localhost:8000`

### 3. Start the React Frontend

```bash
npm start
```

The frontend will open at `http://localhost:3000`

## 🎨 Features

### Modern Chat Interface
- WhatsApp/ChatGPT-style messaging
- Beautiful gradients and glassmorphism effects
- Smooth animations with Framer Motion
- Typing indicators and loading states

### Mathematical Rendering
- LaTeX rendering with KaTeX
- Step-by-step solution breakdown
- Mathematical rule explanations
- Before/after transformations with arrows

### Interactive Graphs
- Embedded Desmos calculator
- Dynamic graph generation
- Responsive iframe embedding
- Error handling for unsupported problems

### Example Problems
Try these examples:
- `Solve: 2x² - 8x + 6 = 0`
- `Find derivative of x³ + 4x² - 2x + 1`
- `Integrate: ∫(3x² + 2x + 1) dx`
- `Factor: x² - 5x + 6`
- `Graph: y = x² - 4x + 3`

## 🏗️ Architecture

### Frontend Stack
- **React 18** - Modern React with hooks
- **TypeScript** - Type safety and better DX
- **Styled Components** - CSS-in-JS styling
- **Framer Motion** - Smooth animations
- **KaTeX** - LaTeX math rendering
- **Axios** - HTTP client for API calls

### Backend Stack
- **FastAPI** - Modern Python API framework
- **MathViz Pipeline** - Mathematical problem solving
- **CORS** - Cross-origin resource sharing
- **Pydantic** - Data validation and serialization

## 📁 Project Structure

```
mathviz-frontend/
├── src/
│   ├── components/          # React components
│   │   ├── MathVizChat.tsx  # Main chat interface
│   │   ├── MessageBubble.tsx # Individual messages
│   │   ├── SolutionDisplay.tsx # Solution breakdown
│   │   └── DesmosGraph.tsx  # Graph component
│   ├── services/            # API services
│   │   └── api.ts           # Backend communication
│   ├── types/               # TypeScript types
│   │   └── mathviz.ts       # MathViz type definitions
│   ├── App.tsx              # Main app component
│   └── index.tsx            # App entry point
├── public/
│   └── index.html           # HTML template
└── package.json             # Dependencies and scripts
```

## 🎛️ Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
REACT_APP_API_URL=http://localhost:8000
```

### Backend Configuration

The backend automatically configures:
- CORS for `localhost:3000`
- MathViz pipeline with enhanced parsing
- Interactive graph generation
- Error handling and validation

## 🔧 Development

### Running Tests
```bash
npm test
```

### Building for Production
```bash
npm run build
```

### Linting and Formatting
```bash
npm run lint
npm run format
```

## 🚢 Deployment

### Frontend Deployment
Build the production bundle:
```bash
npm run build
```

Deploy the `build/` directory to any static hosting service (Netlify, Vercel, etc.)

### Backend Deployment
The FastAPI backend can be deployed to:
- **Heroku** - With Procfile
- **Railway** - Direct deployment
- **DigitalOcean** - App Platform
- **AWS/GCP** - Container deployment

## 🤝 API Integration

The frontend communicates with the backend via REST API:

```typescript
// Solve a problem
const response = await mathvizAPI.solveProblem("2x² - 8x + 6 = 0");

// Get solution with visualization
const response = await mathvizAPI.solveProblemWithVisualization("y = x² + 1");
```

## 🐛 Troubleshooting

### Common Issues

**Backend not connecting:**
- Ensure Python backend is running on port 8000
- Check CORS configuration
- Verify MathViz pipeline is available

**Math rendering issues:**
- KaTeX CSS should be loaded
- Check LaTeX syntax in expressions
- Fallback to code blocks for invalid LaTeX

**Graph not displaying:**
- Verify Desmos URL is valid
- Check iframe CSP policies
- Ensure graph expressions are supported

## 📈 Performance

The frontend is optimized for:
- **Fast loading** - Code splitting and lazy loading
- **Smooth animations** - 60fps animations with Framer Motion
- **Memory efficiency** - Proper cleanup and memoization
- **Network optimization** - Request caching and error retry

## 🎨 Customization

### Theming
Colors and styles can be customized in the styled-components:
- Primary gradient: `#667eea` → `#764ba2`
- Success color: `#10b981`
- Error color: `#ef4444`

### Animation Timing
Framer Motion animations can be adjusted:
- Duration: `0.3s` - `0.6s`
- Easing: `ease-out`, `spring`
- Delays: Staggered by `0.1s`

## 📄 License

This project is part of the MathViz educational tool suite.

---

**Enjoy your beautiful, modern math solving experience! 🧮✨**