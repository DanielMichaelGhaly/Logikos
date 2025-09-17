import React from 'react';
import { createGlobalStyle } from 'styled-components';
import { MathVizChat } from './components/MathVizChat';

const GlobalStyle = createGlobalStyle`
  /* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
  
  html, body {
    height: 100%;
font-family: 'Manrope', -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    overflow: hidden;
    color: #E5E7EB;
    background: radial-gradient(1200px 800px at 20% -10%, #0b2a55 0%, rgba(11,42,85,0.6) 40%, rgba(7,21,43,0.7) 60%),
                linear-gradient(160deg, #07152B 0%, #050F25 50%, #040A1A 100%);
  }
  
  #root {
    height: 100vh;
    overflow: hidden;
  }
  
  /* Custom scrollbar styles */
  ::-webkit-scrollbar {
    width: 10px;
  }
  
  ::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 6px;
  }
  
  ::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 6px;
  }
  
  ::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.25);
  }
  
  /* Selection colors */
  ::selection {
    background: rgba(99, 102, 241, 0.35);
    color: #fff;
  }
  
  /* Focus styles */
  :focus-visible {
    outline: 2px solid rgba(99, 102, 241, 0.6);
    outline-offset: 2px;
  }
  
  /* Prevent text selection on UI elements */
  button, .no-select {
    user-select: none;
  }
  
  /* Smooth transitions */
  * {
    transition: color 0.2s ease, background-color 0.2s ease, border-color 0.2s ease, opacity 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
  }
  
  /* KaTeX fonts */
  .katex {
    font-size: 1em;
  }
  
  /* Improved readability */
  strong, b {
    font-weight: 600;
  }
  
  /* Code styling */
  code {
    font-family: 'JetBrains Mono', 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
  }
`;

const App: React.FC = () => {
  return (
    <>
      <GlobalStyle />
      <MathVizChat />
    </>
  );
};

export default App;