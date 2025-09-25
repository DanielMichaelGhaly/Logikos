import React from 'react';
import { createGlobalStyle } from 'styled-components';
import { ModernMathChat } from './components/ModernMathChat';

const GlobalStyle = createGlobalStyle`
  /* Import Fonts */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  html, body {
    height: 100%;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    overflow: hidden;
    color: #FFFFFF;
    background: #000000;
  }

  #root {
    height: 100vh;
    overflow: hidden;
    background: #000000;
  }

  /* Custom scrollbar styles - dark theme */
  ::-webkit-scrollbar {
    width: 8px;
  }

  ::-webkit-scrollbar-track {
    background: #111111;
  }

  ::-webkit-scrollbar-thumb {
    background: #333333;
    border-radius: 4px;
  }

  ::-webkit-scrollbar-thumb:hover {
    background: #444444;
  }

  /* Selection colors */
  ::selection {
    background: rgba(255, 255, 255, 0.2);
    color: #fff;
  }

  /* Focus styles */
  :focus-visible {
    outline: 2px solid rgba(255, 255, 255, 0.3);
    outline-offset: 2px;
  }

  /* Prevent text selection on UI elements */
  button, .no-select {
    user-select: none;
  }

  /* Smooth transitions */
  * {
    transition: color 0.15s ease, background-color 0.15s ease, border-color 0.15s ease, opacity 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease;
  }

  /* Typography */
  h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF;
    font-weight: 600;
  }

  p {
    color: #E5E5E5;
  }

  /* Code styling */
  code {
    font-family: 'JetBrains Mono', 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
    background: #1A1A1A;
    padding: 2px 6px;
    border-radius: 4px;
    color: #E5E5E5;
  }

  pre {
    background: #1A1A1A;
    padding: 16px;
    border-radius: 8px;
    overflow-x: auto;
    border: 1px solid #333333;
  }

  /* KaTeX styling for dark theme */
  .katex {
    font-size: 1em;
    color: #FFFFFF !important;
  }

  .katex .base {
    color: #FFFFFF !important;
  }
`;

const App: React.FC = () => {
  return (
    <>
      <GlobalStyle />
      <ModernMathChat />
    </>
  );
};

export default App;