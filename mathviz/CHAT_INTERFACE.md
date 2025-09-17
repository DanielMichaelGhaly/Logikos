# 💬 MathViz Chat Interface

A **ChatGPT/WhatsApp-style interface** for solving math problems with **step-by-step solutions** and **Desmos graph integration**.

## 🚀 Quick Start

**Launch the chat interface:**
```bash
cd /Users/sorour/workspace/Logikos
source .venv/bin/activate
python mathviz/run_simple_chat.py
```

Then open your browser to: **http://localhost:8501**

## 💬 How It Works

1. **Type math problems** in the chat input (like messaging apps)
2. **Get instant solutions** with step-by-step breakdowns
3. **View Desmos graphs** for visualizable problems
4. **Continue conversations** with follow-up questions

## 🧮 Example Problems to Try

**Algebra:**
- `Solve for x: 2x + 5 = 13`
- `Find roots of x² - 5x + 6`
- `Factor x² + 7x + 12`

**Calculus:**
- `Find derivative of x² + 3x`
- `Differentiate sin(x) + cos(x)`
- `Integrate 2x + 1`

**Advanced:**
- `Solve system: x + y = 5, 2x - y = 1`
- `Find minimum of x² - 4x + 3`

## 💡 Interface Features

✅ **Chat Bubbles**: Your questions (blue) and MathViz responses (white)
✅ **Typing Indicator**: Shows "MathViz is solving..." with animated dots
✅ **Quick Examples**: Clickable suggestion buttons to get started
✅ **Step Breakdown**: Shows 3 main steps + remaining count
✅ **Desmos Links**: "View Interactive Graph" opens in new tab
✅ **Error Handling**: Friendly error messages for unsupported problems

## 🎯 What You'll See

**You type:** `"Find derivative of x² + 3x"`

**MathViz responds with:**
```
Great! I solved your problem: Find derivative of x² + 3x

📍 Answer: f'(x) = 2x + 3

🔄 Solution Steps:
1. Parse Expression: Extracted expression for differentiation
2. Differentiate: Applied differentiation rules  
3. Simplify: Simplified the result
... and 2 more steps

📊 View Interactive Graph on Desmos
```

## 🔧 Technical Details

- **Backend**: SymPy for accurate mathematical computation
- **Frontend**: Streamlit with custom CSS for chat styling
- **Visualization**: Basic Desmos URL generation for graphs
- **Dependencies**: Only requires existing MathViz components

## 🆚 Comparison with Other Interfaces

| Interface | Style | Graphs | Best For |
|-----------|-------|---------|----------|
| **Chat Interface** | Conversational | Desmos links | Students, quick questions |
| Original Streamlit | Form-based | Interactive plots | Detailed exploration |
| CLI | Command-line | Text output | Scripting, automation |

## 🎨 Visual Design

- **Gradient background** with purple-blue theme
- **Rounded chat bubbles** with shadows
- **Responsive design** works on different screen sizes
- **Clean typography** with proper spacing
- **Color-coded elements**: Green for answers, blue for steps

## 🛠 Troubleshooting

**If you see import errors:**
- Make sure you're in the right directory: `/Users/sorour/workspace/Logikos`
- Activate virtual environment: `source .venv/bin/activate`
- The simple version works without advanced visualization dependencies

**If browser doesn't open automatically:**
- Manually navigate to: `http://localhost:8501`
- Check that port 8501 isn't in use by another app

**If math problems don't solve:**
- Try rephrasing the problem
- Use standard mathematical notation (x^2 instead of x²)
- Check that the problem is supported by SymPy

## 📱 Mobile-Friendly

The interface is designed to work well on:
- ✅ Desktop browsers
- ✅ Tablet screens  
- ✅ Mobile phones (responsive design)

---

**Enjoy your new chat-style math tutor!** 🧮✨