import { createGlobalStyle } from 'styled-components';

export const GlobalStyles = createGlobalStyle`
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #000000;
    color: #FFFFFF;
    line-height: 1.6;
    overflow-x: hidden;
  }

  #root {
    min-height: 100vh;
    background: #000000;
  }

  /* Scrollbar styling for dark theme */
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

  /* Remove default focus outline and add custom one */
  *:focus {
    outline: none;
    box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.2);
  }

  /* Typography */
  h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF;
    font-weight: 600;
  }

  p {
    color: #E5E5E5;
  }

  /* Code blocks */
  code {
    background: #1A1A1A;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    color: #E5E5E5;
  }

  pre {
    background: #1A1A1A;
    padding: 16px;
    border-radius: 8px;
    overflow-x: auto;
    border: 1px solid #333333;
  }
`;