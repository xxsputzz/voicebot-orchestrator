#!/usr/bin/env python3
"""
Prompt Loader Test Suite
Comprehensive testing of the prompt loading and injection system.
"""
import sys
import os
import json
from pathlib import Path

# Add aws_microservices to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "aws_microservices"))

try:
    from prompt_loader import PromptLoader
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the correct directory")
    sys.exit(1)

class PromptLoaderTester:
    """Test suite for PromptLoader functionality"""
    
    def __init__(self):
        """Initialize the tester"""
        self.workspace_root = Path(__file__).parent.parent.parent
        self.docs_path = self.workspace_root / "docs"
        self.prompts_path = self.docs_path / "prompts"
        
        print("📋 Prompt Loader Test Suite")
        print("=" * 50)
    
    def test_prompt_files_exist(self):
        """Test that expected prompt files exist"""
        print(f"\n📁 Testing Prompt Files Existence")
        print("-" * 40)
        
        expected_files = [
            "inbound-call.txt",
            "outbound-call.txt",
            "prompt-main.txt"
        ]
        
        all_exist = True
        
        for filename in expected_files:
            file_path = self.prompts_path / filename
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"  ✅ {filename} exists ({size:,} bytes)")
            else:
                print(f"  ❌ {filename} missing")
                all_exist = False
        
        if all_exist:
            print("  ✅ All expected prompt files found")
        else:
            print("  ⚠️  Some prompt files are missing")
        
        return all_exist
    
    def test_prompt_loader_initialization(self):
        """Test PromptLoader initialization"""
        print(f"\n🔧 Testing PromptLoader Initialization")
        print("-" * 40)
        
        try:
            loader = PromptLoader()
            print("  ✅ PromptLoader initialized successfully")
            
            # Check if it found the prompts directory
            if hasattr(loader, 'prompts_dir'):
                print(f"  ✅ Prompts directory: {loader.prompts_dir}")
            else:
                print("  ⚠️  No prompts directory attribute found")
            
            return loader
            
        except Exception as e:
            print(f"  ❌ PromptLoader initialization failed: {e}")
            return None
    
    def test_prompt_loading(self, loader):
        """Test loading individual prompts"""
        print(f"\n📖 Testing Prompt Loading")
        print("-" * 40)
        
        if not loader:
            print("  ❌ No PromptLoader instance available")
            return False
        
        # Test loading all prompts
        try:
            all_prompts = loader.load_all_prompts()
            if all_prompts:
                print(f"  ✅ Loaded {len(all_prompts)} prompts successfully")
                for prompt_name, content in all_prompts.items():
                    char_count = len(content)
                    line_count = content.count('\n') + 1
                    print(f"    📋 {prompt_name}: {char_count:,} chars, {line_count} lines")
                
                print("  ✅ All prompt loading tests passed")
                return all_prompts
            else:
                print("  ❌ No prompts loaded")
                return False
        except Exception as e:
            print(f"  ❌ Prompt loading failed: {e}")
            return False
    
    def test_prompt_content_quality(self, loaded_prompts):
        """Test the quality and content of loaded prompts"""
        print(f"\n🔍 Testing Prompt Content Quality")
        print("-" * 40)
        
        if not loaded_prompts:
            print("  ❌ No loaded prompts available for testing")
            return False
        
        quality_checks = []
        
        # Check main prompt content
        if "prompt-main" in loaded_prompts:
            main_content = loaded_prompts["prompt-main"]
            
            # Check for key terms
            main_keywords = [
                "assistant", "help", "respond", "professional",
                "context", "information", "customer"
            ]
            
            found_keywords = []
            for keyword in main_keywords:
                if keyword.lower() in main_content.lower():
                    found_keywords.append(keyword)
            
            if len(found_keywords) >= 4:  # At least 4 out of 7 general terms
                print(f"  ✅ Main prompt contains {len(found_keywords)}/7 relevant keywords")
                quality_checks.append(True)
            else:
                print(f"  ⚠️  Main prompt only contains {len(found_keywords)}/7 relevant keywords")
                quality_checks.append(False)
        
        # Check inbound call prompt
        if "inbound-call" in loaded_prompts:
            inbound_content = loaded_prompts["inbound-call"]
            
            inbound_keywords = ["inbound", "customer", "call", "help", "service"]
            found_inbound = sum(1 for kw in inbound_keywords if kw.lower() in inbound_content.lower())
            
            if found_inbound >= 3:
                print(f"  ✅ Inbound prompt contains {found_inbound}/5 relevant keywords")
                quality_checks.append(True)
            else:
                print(f"  ⚠️  Inbound prompt only contains {found_inbound}/5 relevant keywords")
                quality_checks.append(False)
        
        # Check outbound call prompt
        if "outbound-call" in loaded_prompts:
            outbound_content = loaded_prompts["outbound-call"]
            
            outbound_keywords = ["outbound", "contact", "call", "reach", "follow"]
            found_outbound = sum(1 for kw in outbound_keywords if kw.lower() in outbound_content.lower())
            
            if found_outbound >= 2:
                print(f"  ✅ Outbound prompt contains {found_outbound}/5 relevant keywords")
                quality_checks.append(True)
            else:
                print(f"  ⚠️  Outbound prompt only contains {found_outbound}/5 relevant keywords")
                quality_checks.append(False)
        
        # Overall quality assessment
        passed_checks = sum(quality_checks)
        total_checks = len(quality_checks)
        
        if passed_checks == total_checks:
            print("  ✅ All prompt content quality checks passed")
            return True
        else:
            print(f"  ⚠️  {passed_checks}/{total_checks} quality checks passed")
            return passed_checks > 0
    
    def test_prompt_combination(self, loader):
        """Test combining multiple prompts"""
        print(f"\n🔗 Testing Prompt Combination")
        print("-" * 40)
        
        if not loader:
            print("  ❌ No PromptLoader instance available")
            return False
        
        try:
            # Test system prompt generation with inbound call type
            inbound_prompt = loader.get_system_prompt(call_type="inbound")
            
            if inbound_prompt:
                char_count = len(inbound_prompt)
                print(f"  ✅ Inbound system prompt generated: {char_count:,} characters")
                
                # Check that it contains inbound-specific content
                if "inbound" in inbound_prompt.lower():
                    print("  ✅ Inbound prompt contains expected call-type content")
                else:
                    print("  ⚠️  Inbound prompt may not contain call-type specific content")
            else:
                print("  ❌ No inbound system prompt generated")
                return False
            
            # Test system prompt generation with outbound call type
            outbound_prompt = loader.get_system_prompt(call_type="outbound")
            
            if outbound_prompt:
                char_count = len(outbound_prompt)
                print(f"  ✅ Outbound system prompt generated: {char_count:,} characters")
                
                # Check that it contains outbound-specific content
                if "outbound" in outbound_prompt.lower():
                    print("  ✅ Outbound prompt contains expected call-type content")
                else:
                    print("  ⚠️  Outbound prompt may not contain call-type specific content")
            else:
                print("  ❌ No outbound system prompt generated")
                return False
            
            # Test general system prompt (no call type)
            general_prompt = loader.get_system_prompt()
            
            if general_prompt:
                char_count = len(general_prompt)
                print(f"  ✅ General system prompt generated: {char_count:,} characters")
                print("  ✅ Prompt combination test passed")
                return True
            else:
                print("  ❌ No general system prompt generated")
                return False
                
        except Exception as e:
            print(f"  ❌ Prompt combination test failed: {e}")
            return False
    
    def test_prompt_caching(self, loader):
        """Test prompt caching functionality"""
        print(f"\n💾 Testing Prompt Caching")
        print("-" * 40)
        
        if not loader:
            print("  ❌ No PromptLoader instance available")
            return False
        
        try:
            import time
            
            # Load prompts multiple times and measure performance
            # First load (should cache)
            start_time = time.time()
            first_load = loader.load_all_prompts()
            first_time = time.time() - start_time
            
            # Second load (should use cache)
            start_time = time.time()
            second_load = loader.load_all_prompts()
            second_time = time.time() - start_time
            
            if first_load and second_load:
                if first_load == second_load:
                    print(f"  ✅ Content consistency: First and second load identical")
                    print(f"  📊 Performance: First load: {first_time:.4f}s, Second load: {second_time:.4f}s")
                    
                    # Cache should make second load significantly faster (or at least not slower)
                    if second_time <= first_time * 2:  # Allow some variance
                        print("  ✅ Caching appears to be working (second load not slower)")
                    else:
                        print("  ⚠️  Second load slower than expected (caching may not be working)")
                    
                    # Test force reload
                    start_time = time.time()
                    reload_result = loader.reload_prompts()
                    reload_time = time.time() - start_time
                    
                    if reload_result == first_load:
                        print(f"  ✅ Force reload works: {reload_time:.4f}s")
                        return True
                    else:
                        print("  ❌ Force reload returned different content")
                        return False
                else:
                    print("  ❌ Content mismatch between first and second load")
                    return False
            else:
                print("  ❌ Failed to load prompts for caching test")
                return False
                
        except Exception as e:
            print(f"  ❌ Prompt caching test failed: {e}")
            return False
    
    def test_error_handling(self, loader):
        """Test error handling for invalid prompts"""
        print(f"\n⚠️  Testing Error Handling")
        print("-" * 40)
        
        if not loader:
            print("  ❌ No PromptLoader instance available")
            return False
        
        # Test available prompts method
        try:
            available = loader.get_available_prompts()
            print(f"  ✅ Available prompts: {available}")
        except Exception as e:
            print(f"  ❌ Error getting available prompts: {e}")
            return False
        
        # Test system prompt with invalid call type
        try:
            invalid_prompt = loader.get_system_prompt(call_type="invalid_type")
            if invalid_prompt:
                print("  ✅ Invalid call type handled gracefully (returned default prompt)")
            else:
                print("  ✅ Invalid call type handled gracefully (returned empty)")
        except Exception as e:
            print(f"  ❌ Invalid call type caused exception: {e}")
            return False
        
        # Test system prompt with non-existent include files
        try:
            missing_files_prompt = loader.get_system_prompt(include_files=["nonexistent-prompt"])
            if missing_files_prompt == "" or missing_files_prompt is None:
                print("  ✅ Non-existent include files handled gracefully")
            else:
                print("  ⚠️  Non-existent include files returned content (may be fallback)")
        except Exception as e:
            print(f"  ❌ Non-existent include files caused exception: {e}")
            return False
        
        print("  ✅ Error handling tests passed")
        return True
    
    def run_all_tests(self):
        """Run all prompt loader tests"""
        print("🚀 Starting Complete Prompt Loader Test Suite")
        print("=" * 60)
        
        test_results = []
        
        # Test 1: Check prompt files exist
        result1 = self.test_prompt_files_exist()
        test_results.append(("Prompt Files Existence", result1))
        
        # Test 2: PromptLoader initialization
        loader = self.test_prompt_loader_initialization()
        result2 = bool(loader)
        test_results.append(("PromptLoader Initialization", result2))
        
        if not loader:
            print("⚠️  Cannot continue with remaining tests without PromptLoader")
        else:
            # Test 3: Prompt loading
            loaded_prompts = self.test_prompt_loading(loader)
            result3 = bool(loaded_prompts)
            test_results.append(("Prompt Loading", result3))
            
            # Test 4: Content quality
            result4 = self.test_prompt_content_quality(loaded_prompts) if loaded_prompts else False
            test_results.append(("Prompt Content Quality", result4))
            
            # Test 5: Prompt combination
            result5 = self.test_prompt_combination(loader)
            test_results.append(("Prompt Combination", result5))
            
            # Test 6: Prompt caching
            result6 = self.test_prompt_caching(loader)
            test_results.append(("Prompt Caching", result6))
            
            # Test 7: Error handling
            result7 = self.test_error_handling(loader)
            test_results.append(("Error Handling", result7))
        
        # Print test summary
        print(f"\n📊 Test Results Summary")
        print("=" * 60)
        
        passed = 0
        for test_name, result in test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"  {test_name:<30} {status}")
            if result:
                passed += 1
        
        total_tests = len(test_results)
        print(f"\n🎯 Overall Result: {passed}/{total_tests} tests passed")
        
        if passed == total_tests:
            print("🎉 All prompt loader tests passed!")
        else:
            print(f"⚠️  {total_tests - passed} tests failed")
        
        return passed == total_tests

def main():
    """Main test execution"""
    try:
        tester = PromptLoaderTester()
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
