"""
Test script for the new simplified Logikos architecture.
Tests the complete pipeline: classification → dual solving → confidence comparison.
"""

import sys
sys.path.append('.')

from backend.input_processor.classifier import QuestionClassifier
from backend.solvers.sympy_solver import SympySolver
from backend.solvers.ai_reasoner import AIReasoner
from backend.solvers.confidence import ConfidenceComparator


def test_complete_pipeline():
    """Test the complete new architecture pipeline."""
    print("🚀 Testing New Logikos Architecture")
    print("=" * 50)

    # Initialize components
    print("Initializing components...")
    classifier = QuestionClassifier()
    sympy_solver = SympySolver()
    ai_reasoner = AIReasoner()
    confidence_comparator = ConfidenceComparator()

    # Test cases
    test_questions = [
        "solve 2x + 5 = 0",
        "factor x^2 - 4",
        "derivative of x^2 + 3x",
        "plot sin(x) from 0 to 2π"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Test {i}: {question}")
        print("-" * 30)

        try:
            # Step 1: Classification
            print("🔍 Classifying question...")
            classification = classifier.classify_question(question)
            print(f"Type: {classification.type}")
            print(f"Expression: {classification.expression}")
            print(f"Visualization hint: {classification.visualization_hint}")

            # Step 2: SymPy solving
            print("\n🔢 SymPy solving...")
            sympy_result = sympy_solver.solve_question(classification)
            if sympy_result.success:
                print(f"Result: {sympy_result.result}")
                print(f"Steps: {len(sympy_result.steps)}")
            else:
                print(f"Error: {sympy_result.error}")

            # Step 3: AI reasoning (if available)
            print("\n🤖 AI reasoning...")
            ai_result = ai_reasoner.reason_about_question(classification)
            if ai_result.success:
                print(f"Explanation length: {len(ai_result.explanation or '')}")
                print(f"Mathematical result: {ai_result.mathematical_result}")
                print(f"Representable: {ai_result.representable}")
            else:
                print(f"AI Error: {ai_result.error}")

            # Step 4: Confidence comparison
            print("\n⚖️ Confidence comparison...")
            confidence = confidence_comparator.compare_results(
                classification, sympy_result, ai_result
            )
            print(f"Confidence level: {confidence.confidence_level}")
            print(f"Show to user: {confidence.show_to_user}")
            if confidence.comparison_details:
                print(f"Details: {confidence.comparison_details}")

        except Exception as e:
            print(f"❌ Error in test {i}: {str(e)}")

        print("\n" + "=" * 50)

    print("✅ Architecture test completed!")


def test_individual_components():
    """Test individual components."""
    print("\n🔧 Testing Individual Components")
    print("=" * 50)

    # Test classifier only
    print("Testing classifier...")
    classifier = QuestionClassifier()
    result = classifier.classify_question("solve x^2 - 9 = 0")
    print(f"Classification: {result.type}, Expression: {result.expression}")

    # Test SymPy solver only
    print("\nTesting SymPy solver...")
    sympy_solver = SympySolver()
    sympy_result = sympy_solver.solve_question(result)
    print(f"SymPy success: {sympy_result.success}, Result: {sympy_result.result}")

    print("✅ Individual component tests completed!")


if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Complete pipeline test")
    print("2. Individual components test")
    print("3. Both")

    choice = input("Enter choice (1/2/3): ").strip()

    if choice in ["1", "3"]:
        test_complete_pipeline()

    if choice in ["2", "3"]:
        test_individual_components()

    print("\n🎯 New architecture is ready for testing!")
    print("Next steps:")
    print("- Test with Ollama integration")
    print("- Add visualization generation")
    print("- Build React frontend integration")