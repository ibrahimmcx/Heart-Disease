/* -------------------------------------------------------------
   CARDIO-SHIELD — Runtime Configuration

   Bu dosyada backend API URL'ini tanımlayın.

   - LOCAL DEV : boş string bırakın ("") → aynı origin kullanılır
   - PRODUCTION: Render servis URL'inizi buraya yazın
   ------------------------------------------------------------- */

const APP_CONFIG = {
    // Render servis URL (boş bırakırsanız Vercel same-origin kullanılır)
    RENDER_API_URL: "https://heart-disease-gcsy.onrender.com"
};
