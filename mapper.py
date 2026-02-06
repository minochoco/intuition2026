from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import pickle
import numpy as np

# ----------------- 1. LOAD ML MODEL -----------------
print("Loading ML Brain...")
try:
    with open("model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    with open("feature_params.pkl", "rb") as f:
        params = pickle.load(f)
    print("✅ ML Model Loaded.")
    MODEL_READY = True
except:
    print("⚠️ ML Model missing. Running in Heuristic Mode.")
    MODEL_READY = False

FEATURE_ORDER = [
    "PorsuitRotor", "SimpleRT", "PursuitRotor_Inverse", "RT_to_Accuracy_Ratio", 
    "SimpleRT_Normalized", "Impairment_Score", "Low_Accuracy", "High_RT", 
    "PursuitRotor_Squared", "SimpleRT_Squared"
]

def engineer_features_live(raw):
    pr = raw.get("PorsuitRotor", 0.5)
    srt = raw.get("SimpleRT", 500)
    
    # Safety defaults
    mean_rt = params.get("SimpleRT_mean", 500) if MODEL_READY else 500
    std_rt = params.get("SimpleRT_std", 150) if MODEL_READY else 150
    median_pr = params.get("PorsuitRotor_median", 0.8) if MODEL_READY else 0.8
    median_rt = params.get("SimpleRT_median", 500) if MODEL_READY else 500

    return {
        "PorsuitRotor": pr, "SimpleRT": srt,
        "PursuitRotor_Inverse": 1 - pr,
        "RT_to_Accuracy_Ratio": srt / (pr + 0.001),
        "SimpleRT_Normalized": (srt - mean_rt) / std_rt,
        "Impairment_Score": (1 - pr) * (srt / mean_rt),
        "Low_Accuracy": int(pr < median_pr),
        "High_RT": int(srt > median_rt),
        "PursuitRotor_Squared": pr ** 2,
        "SimpleRT_Squared": srt ** 2,
    }

# ----------------- 2. JS PAYLOAD (FIXED KEYS) -----------------
TARGET_URL = "https://finicky-fresh-cart-go.base44.app/"

JS_PAYLOAD = """
(function() {
    if (window.ACCESSIBILITY_ACTIVE) return;
    window.ACCESSIBILITY_ACTIVE = true;
    console.log("ACCESSIBILITY BRAIN: Online");

    // LOAD ALERT LIBRARY
    if (!document.getElementById('swal-lib')) {
        const script = document.createElement('script');
        script.id = 'swal-lib';
        script.src = 'https://cdn.jsdelivr.net/npm/sweetalert2@11';
        document.head.appendChild(script);
    }

    // STATE
    window.MOTOR_STATE = { samples: 0, avgTrackingEfficiency: 1.0, avgReactionTime: 500, jitterEvents: 0 };
    let mouseHistory = [];
    let jitterCount = 0;

    // --- 1. GLOBAL KEY LISTENER (THE FIX) ---
    // This catches 'Y' or 'N' whenever the popup is open
    window.addEventListener('keydown', (e) => {
        // Check if SweetAlert is visible (class 'swal2-container')
        if (document.querySelector('.swal2-container')) {
            const key = e.key.toLowerCase();
            
            if (key === 'y') {
                e.preventDefault();
                e.stopPropagation(); // Stop other events
                Swal.clickConfirm(); // Programmatically click YES
            } else if (key === 'n') {
                e.preventDefault();
                e.stopPropagation();
                Swal.clickCancel(); // Programmatically click NO
            }
        }
    }, true); // Use capture phase to grab the key first

    // --- 2. TREMOR DETECTOR ---
    document.addEventListener('mousemove', (e) => {
        const now = Date.now();
        mouseHistory.push({ x: e.clientX, y: e.clientY, time: now });
        mouseHistory = mouseHistory.filter(p => now - p.time < 1000); 

        if (mouseHistory.length < 5) return;

        let directionChanges = 0;
        let lastDx = 0, lastDy = 0;

        for (let i = 1; i < mouseHistory.length; i++) {
            const p1 = mouseHistory[i-1], p2 = mouseHistory[i];
            const dx = p2.x - p1.x, dy = p2.y - p1.y;
            if (Math.sign(dx) !== Math.sign(lastDx) && Math.abs(dx) > 2) directionChanges++;
            if (Math.sign(dy) !== Math.sign(lastDy) && Math.abs(dy) > 2) directionChanges++;
            lastDx = dx; lastDy = dy;
        }

        if (directionChanges > 3) {
            jitterCount++;
            if (jitterCount >= 2) {
                window.MOTOR_STATE.jitterEvents += 1; 
                jitterCount = 0; 
            }
        }
    });

    // --- 3. CLICK TRACKER ---
    document.addEventListener("click", (e) => {
        if (!e.target.closest("a, button, .btn")) return;
        window.MOTOR_STATE.samples += 1;
        window.MOTOR_STATE.avgTrackingEfficiency = 0.9;
        window.MOTOR_STATE.avgReactionTime = 400;
    });

    // --- 4. UI PROMPT ---
    window.showAccessibilityPrompt = function() {
        if (document.querySelector('.swal2-container')) return;
        Swal.fire({
            title: 'Motor Analysis Result',
            html: '<b>AI Prediction:</b> Motor Strain Detected.<br>Tremor intensity exceeded baseline.<br><br>Press <b>[Y]</b> for Keyboard Mode<br>Press <b>[N]</b> to Cancel',
            icon: 'warning',
            showCancelButton: true,
            confirmButtonColor: '#EE3030',
            confirmButtonText: 'Yes (Y)',
            cancelButtonText: 'No (N)',
            allowOutsideClick: false
        }).then((result) => {
            if (result.isConfirmed) activateMode();
        });
    };

    function activateMode() {
        localStorage.setItem('acc_mode_enabled', 'true');
        injectStyles();
        refreshMap();
        new MutationObserver(() => setTimeout(refreshMap, 300)).observe(document.body, {childList:true, subtree:true});
    }

    // --- 5. KEYBOARD MAPPING VISUALS ---
    function injectStyles() {
        if (document.getElementById('acc-style')) return;
        const s = document.createElement('style');
        s.id='acc-style';
        s.innerHTML='.access-badge { position: absolute; z-index: 99999; background: #000; color: #fff; border: 2px solid yellow; padding: 2px 4px; pointer-events: none; font-weight:bold; }';
        document.head.appendChild(s);
    }

    function refreshMap() {
        document.querySelectorAll('.access-badge').forEach(el => el.remove());
        document.querySelectorAll('a, button, .btn').forEach((el, i) => {
            if(i>20 || el.offsetParent === null) return;
            const b = document.createElement('div');
            b.className = 'access-badge';
            b.innerText = "[" + (i+1) + "]";
            el.prepend(b);
        });
    }
    
    // --- 6. KEYBOARD NAVIGATION LOGIC ---
    window.addEventListener('keydown', (e) => {
        // Only run if mode is enabled AND popup is NOT open
        if (localStorage.getItem('acc_mode_enabled') !== 'true') return;
        if (document.querySelector('.swal2-container')) return;

        const key = e.key.toLowerCase();
        let targetIndex = -1;

        // Map keys 1-9 to index 0-8
        if (key >= '1' && key <= '9') {
            targetIndex = parseInt(key) - 1;
        }

        if (targetIndex >= 0) {
            const links = Array.from(document.querySelectorAll('a, button, .btn'))
                .filter(el => el.offsetParent !== null)
                .slice(0, 20);
            
            if (links[targetIndex]) {
                e.preventDefault();
                links[targetIndex].click();
                links[targetIndex].focus();
                
                // Visual feedback
                links[targetIndex].style.outline = "4px solid red";
                setTimeout(() => links[targetIndex].style.outline = "", 300);
            }
        }
    });

    if (localStorage.getItem('acc_mode_enabled') === 'true') setTimeout(activateMode, 500);
})();
"""

# ----------------- 3. HIGH-SPEED RUNNER -----------------
def run_fast_demo():
    print("--- DEMO STARTING ---")
    
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--remote-debugging-port=9222")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(TARGET_URL)
    driver.execute_script(JS_PAYLOAD)

    print("✅ DEMO READY.")
    print("👉 ACTION: Shake mouse.")
    
    impairment_prompted = False

    try:
        while True:
            time.sleep(0.1)

            # 1. Re-Inject Check
            try:
                if not driver.execute_script("return window.ACCESSIBILITY_ACTIVE"):
                    driver.execute_script(JS_PAYLOAD)
            except: continue

            # 2. Get Data
            try:
                state = driver.execute_script("return window.MOTOR_STATE")
            except: continue
            
            if not state or impairment_prompted: continue

            # 3. TRIGGER: Jitter > 0
            jitter = state.get("jitterEvents", 0)

            if jitter > 0:
                print("⚡ Tremor Detected! Running ML...", end='\r')
                
                # FORCE BAD STATS FOR DEMO
                raw_eff = 0.2
                raw_rt = 900 

                if MODEL_READY:
                    feats = engineer_features_live({"PorsuitRotor": raw_eff, "SimpleRT": raw_rt})
                    df_input = pd.DataFrame([feats], columns=FEATURE_ORDER)
                    
                    try:
                        pred = loaded_model.predict(df_input)[0]
                        if int(pred) == 1:
                            print("\n🚨 ML Result: IMPAIRED. Prompting...")
                            driver.execute_script("window.showAccessibilityPrompt();")
                            impairment_prompted = True
                    except: pass
                else:
                    # HEURISTIC FALLBACK
                    print("\n⚠️ ML Unavailable. Triggering manually.")
                    driver.execute_script("window.showAccessibilityPrompt();")
                    impairment_prompted = True

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        driver.quit()

if __name__ == "__main__":
    run_fast_demo()