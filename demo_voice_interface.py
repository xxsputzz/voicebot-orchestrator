#!/usr/bin/env python3

# Demo of the improved voice selection interface

def demo_voice_selection():
    voices = [
        "angie", "denise", "freeman", "geralt", "halle", "jlaw", 
        "lj", "myself", "pat", "pat2", "rainbow", "snakes", 
        "train_dotcom", "train_daws", "train_dreams", "train_grace",
        "train_lescault", "train_mouse", "william", "random",
        "emma", "sophia", "olivia", "isabella", "mia", "charlotte", 
        "ava", "amelia", "harper", "evelyn"
    ]
    
    print("\n🎭 VOICE SELECTION")
    print("=" * 60)
    
    # Display voices in two columns, simple format
    print(f"{'COLUMN 1':<28} {'COLUMN 2':<28}")
    print("-" * 56)
    
    mid_point = (len(voices) + 1) // 2
    
    for i in range(mid_point):
        # Left column
        if i < len(voices):
            left_voice = voices[i]
            left_text = f"{i+1:2d}. {left_voice}"
        else:
            left_text = ""
        
        # Right column
        right_index = i + mid_point
        if right_index < len(voices):
            right_voice = voices[right_index]
            right_text = f"{right_index+1:2d}. {right_voice}"
        else:
            right_text = ""
        
        print(f"{left_text:<28} {right_text:<28}")
    
    print(f"\n   0. Use default voice (angie)")
    print(f"\nTotal voices: {len(voices)}")
    
    print("\n" + "="*60)
    print("Sample synthesis parameters:")
    print("📊 Synthesis Parameters:")
    print("   Voice: emma")
    print("   Quality: high_quality") 
    print("   Text length: 50 characters")
    print("   Estimated words: 8")
    print("")
    print("❓ Proceed with synthesis?")
    print("   Type 'yes' or 'y' to continue, anything else to cancel: [WAITING FOR USER INPUT]")

if __name__ == "__main__":
    demo_voice_selection()
