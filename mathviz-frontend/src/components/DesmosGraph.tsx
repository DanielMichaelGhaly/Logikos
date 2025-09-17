import React, { useEffect, useRef } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

const GraphContainer = styled(motion.div)`
  margin: 1.5rem 0;
  background: white;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(0, 0, 0, 0.05);
`;

const GraphHeader = styled.div`
  background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.06));
  color: #E5E7EB;
  padding: 1rem 1.5rem;
  font-weight: 700;
  display: flex;
  align-items: center;
`;

const GraphCanvas = styled.div`
  width: 100%;
  height: 400px;
  display: block;
`;

const ErrorMessage = styled.div`
  padding: 2rem;
  text-align: center;
  color: #6b7280;
  font-style: italic;
`;

interface DesmosGraphProps {
  url?: string; // optional iframe fallback
  expressions?: string[];
  viewport?: { xmin?: number; xmax?: number; ymin?: number; ymax?: number };
}

// Desmos is loaded from https://www.desmos.com/api/v1.8/calculator.js via index.html or dynamically if needed

export const DesmosGraph: React.FC<DesmosGraphProps> = ({ url, expressions, viewport }) => {
  const elRef = useRef<HTMLDivElement | null>(null);
  const calculatorRef = useRef<any>(null);

  useEffect(() => {
    const hasExpressions = expressions && expressions.length > 0;
    if (!hasExpressions || !elRef.current) return;

    // Ensure Desmos is available
    const ensureDesmos = async () => {
      if ((window as any).Desmos) return (window as any).Desmos;
      await new Promise<void>((resolve, reject) => {
        const s = document.createElement('script');
        s.src = 'https://www.desmos.com/api/v1.8/calculator.js?apiKey=desa';
        s.async = true;
        s.onload = () => resolve();
        s.onerror = () => reject(new Error('Failed to load Desmos API'));
        document.body.appendChild(s);
      });
      return (window as any).Desmos;
    };

    let disposed = false;
    ensureDesmos()
      .then((Desmos) => {
        if (disposed || !elRef.current) return;
        calculatorRef.current = Desmos.GraphingCalculator(elRef.current, {
          expressions: true,
          settingsMenu: false,
          zoomButtons: true,
          expressionsCollapsed: false,
        });

        // Set expressions
        expressions!.forEach((expr) => {
          try {
            calculatorRef.current!.setExpression({ id: String(Math.random()), latex: expr });
          } catch {
            // ignore invalid expression
          }
        });

        // Set viewport if provided
        if (viewport) {
          const { xmin, xmax, ymin, ymax } = viewport;
          if ([xmin, xmax, ymin, ymax].some((v) => typeof v === 'number')) {
            calculatorRef.current.setMathBounds({
              left: xmin ?? -10,
              right: xmax ?? 10,
              bottom: ymin ?? -10,
              top: ymax ?? 10,
            });
          }
        }
      })
      .catch(() => {})

    return () => {
      disposed = true;
      if (calculatorRef.current) {
        try { calculatorRef.current.destroy(); } catch {}
        calculatorRef.current = null;
      }
    };
  }, [expressions, viewport]);

  // If config is provided, render canvas; otherwise if URL given, show iframe fallback
  if (expressions && expressions.length > 0) {
    return (
      <GraphContainer initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.5 }}>
        <GraphHeader>Interactive Graph</GraphHeader>
        <GraphCanvas ref={elRef} />
      </GraphContainer>
    );
  }

  if (url) {
    return (
      <GraphContainer initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.5 }}>
        <GraphHeader>Interactive Graph</GraphHeader>
        <iframe src={url} title="Interactive Mathematical Graph" style={{ width: '100%', height: 400, border: 'none', display: 'block' }} />
      </GraphContainer>
    );
  }

  return (
    <GraphContainer initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.5 }}>
      <GraphHeader>Interactive Graph</GraphHeader>
      <ErrorMessage>Graph visualization not available for this problem.</ErrorMessage>
    </GraphContainer>
  );
};
