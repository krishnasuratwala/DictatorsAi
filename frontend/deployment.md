# Frontend Deployment (Vercel/Netlify)

This folder contains the React application. 

**Recommended Host**: Vercel (Fastest & Easiest) or Netlify.

## Option A: Deploy to Vercel (Recommended)
1.  Push this code to GitHub.
2.  Go to [Vercel](https://vercel.com).
3.  Click **"Add New Project"** -> **"Import"** your repository.
4.  **Root Directory**: Click "Edit" and select `frontend`.
5.  **Build Command**: `npm run build`
6.  **Output Directory**: `dist`
7.  **Environment Variables**:
    *   `VITE_API_URL`: Set this to your Render Middleware URL (e.g., `https://dictator-ai-middleware.onrender.com`).
    *   *Note*: You need to update `App.tsx` to use `import.meta.env.VITE_API_URL` instead of relative paths if you do this separation!

> [!IMPORTANT]
> **Do I need to upload the `dist` folder?**
> **NO.** You do NOT need to create or upload the `dist` folder manually. Vercel will run `npm run build` on their servers and generate it automatically.
