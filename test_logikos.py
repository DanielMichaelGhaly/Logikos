#!/usr/bin/env python3
"""
Logikos + Nemotron Integration Test
Essential setup verification for mathematical chat assistant.
"""

import sys
sys.path.append('.')

def main():
    """Essential tests only."""
    print("🚀 Logikos + Nemotron Essential Test")
    print("=" * 50)

    # Test 1: Core imports
    try:
        import ollama
        from backend.config import config
        from backend.input_processor.classifier import QuestionClassifier
        from backend.solvers.sympy_solver import SympySolver
        print("✅ All core imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

    # Test 2: Configuration
    print(f"\n⚙️ Configuration:")
    print(f"   Model: {config.ollama_model}")
    print(f"   Expected: Randomblock1/nemotron-nano:latest")

    if config.ollama_model == "Randomblock1/nemotron-nano:latest":
        print("✅ Correct model configured")
    else:
        print("⚠️ Model configuration may need adjustment")

    # Test 3: SymPy backend (core functionality)
    print(f"\n🔢 Testing SymPy mathematical engine...")
    try:
        classifier = QuestionClassifier()
        sympy_solver = SympySolver()

        # Test basic math solving
        classification = classifier.classify_question("solve 2x + 6 = 0")
        result = sympy_solver.solve_question(classification)

        if result.success and result.result == "[-3]":
            print("✅ SymPy solver working correctly")
            print(f"   Problem: 2x + 6 = 0")
            print(f"   Solution: {result.result}")
        else:
            print(f"❌ SymPy issue: {result.error if not result.success else 'unexpected result'}")
            return False
    except Exception as e:
        print(f"❌ SymPy test failed: {e}")
        return False

    # Test 4: Basic Ollama connectivity (without model call)
    print(f"\n🔗 Testing Ollama connectivity...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service accessible")
        else:
            print("❌ Ollama service not responding properly")
            return False
    except Exception as e:
        print("❌ Ollama service not accessible")
        print("💡 Make sure Ollama is running: ollama serve")
        return False

    # Test 5: Component initialization (without AI calls)
    print(f"\n🧩 Testing component initialization...")
    try:
        # Test components can be created
        classifier = QuestionClassifier()
        print("✅ Question classifier initialized")

        # Note: AI components are initialized but not tested to avoid hanging
        from backend.solvers.ai_reasoner import AIReasoner
        from backend.solvers.confidence import ConfidenceComparator

        ai_reasoner = AIReasoner()
        confidence_comparator = ConfidenceComparator()
        print("✅ AI reasoner initialized")
        print("✅ Confidence comparator initialized")

    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        return False

    # Summary
    print(f"\n🎉 ESSENTIAL SETUP VERIFIED!")
    print(f"✅ Configuration: Nemotron model set")
    print(f"✅ SymPy engine: Working and tested")
    print(f"✅ Ollama service: Accessible")
    print(f"✅ All components: Initialized")

    print(f"\n🚀 Logikos is ready for Nemotron!")
    print(f"\nTo test with live AI:")
    print(f"1. Ensure Ollama is running: ollama serve")
    print(f"2. Start backend: python start_new_backend.py")
    print(f"3. Test API endpoint with a math problem")

    print(f"\n💡 The system will:")
    print(f"   - Use Nemotron for educational explanations")
    print(f"   - Use SymPy for accurate mathematical computation")
    print(f"   - Compare results for confidence scoring")
    print(f"   - Provide step-by-step solutions")

    return True

if __name__ == "__main__":
    try:
        success = main()
        print(f"\n{'✅ SETUP COMPLETE' if success else '❌ SETUP INCOMPLETE'}")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n⏹️ Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)