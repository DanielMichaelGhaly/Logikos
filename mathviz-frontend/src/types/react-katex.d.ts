declare module 'react-katex' {
  import * as React from 'react';

  export interface KatexBaseProps {
    children: string;
  }

  export const InlineMath: React.FC<KatexBaseProps>;
  export const BlockMath: React.FC<KatexBaseProps>;
}
