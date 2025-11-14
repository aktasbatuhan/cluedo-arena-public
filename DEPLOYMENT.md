# Deployment Guide - Cluedo Arena

This guide will help you deploy the Cluedo Arena application to various cloud platforms.

## Table of Contents
- [Platform Comparison](#platform-comparison)
- [Railway Deployment (Recommended)](#railway-deployment-recommended)
- [Render Deployment](#render-deployment)
- [Vercel Deployment (Limited)](#vercel-deployment-limited)
- [Environment Variables](#environment-variables)
- [Local Testing](#local-testing)

---

## Platform Comparison

| Platform | WebSocket Support | Pricing | Ease of Deploy | Recommendation |
|----------|------------------|---------|----------------|----------------|
| **Railway** | ✅ Native | Free tier available | ⭐⭐⭐⭐⭐ | **Best Choice** |
| **Render** | ✅ Native | Free tier available | ⭐⭐⭐⭐ | Great alternative |
| **Vercel** | ⚠️ Limited | Free tier available | ⭐⭐ | Not recommended |

**Why Railway/Render?** This application uses Socket.IO for real-time communication. Vercel's serverless architecture doesn't support persistent WebSocket connections natively, making Railway or Render better choices.

---

## Railway Deployment (Recommended)

Railway provides excellent Socket.IO support with zero configuration needed.

### Prerequisites
- GitHub account
- Railway account (sign up at [railway.app](https://railway.app))

### Deployment Steps

1. **Connect GitHub Repository**
   - Go to [railway.app](https://railway.app)
   - Click "Start a New Project"
   - Select "Deploy from GitHub repo"
   - Authorize Railway to access your repositories
   - Select the `cluedo-arena-public` repository

2. **Configure Build Settings**
   - Railway will auto-detect the Node.js project
   - Root directory: `/cluedo_game_engine`
   - Build command: `npm install`
   - Start command: `npm start`

3. **Set Environment Variables**

   Go to your project's Variables section and add:

   ```bash
   NODE_ENV=production
   PORT=8080
   LLM_BACKEND=OPENROUTER  # or COHERE or PREDIBASE

   # Add your API keys (at least one required):
   OPENROUTER_API_KEY=your_openrouter_key_here
   COHERE_API_KEY=your_cohere_key_here
   PREDIBASE_API_KEY=your_predibase_key_here
   ```

4. **Deploy**
   - Railway will automatically deploy your application
   - Once deployed, you'll get a public URL like `your-app.railway.app`
   - The app will auto-deploy on every push to your main branch

5. **Access Your App**
   - Click the generated domain to access your Cluedo Arena
   - Start playing! 🎮

### Railway Configuration File

The `railway.json` file in the project root configures:
- Build process
- Start command
- Restart policy

---

## Render Deployment

Render is another excellent choice for Socket.IO applications.

### Prerequisites
- GitHub account
- Render account (sign up at [render.com](https://render.com))

### Deployment Steps

1. **Create New Web Service**
   - Go to [render.com/dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the `cluedo-arena-public` repository

2. **Configure Service**

   ```
   Name: cluedo-arena
   Root Directory: cluedo_game_engine
   Environment: Node
   Region: Choose closest to your users
   Branch: main (or your default branch)
   Build Command: npm install
   Start Command: npm start
   ```

3. **Select Instance Type**
   - Free tier: Spins down after inactivity (cold starts)
   - Starter ($7/month): Always on, better performance

4. **Set Environment Variables**

   Click "Advanced" and add these environment variables:

   ```bash
   NODE_ENV=production
   PORT=8080
   LLM_BACKEND=OPENROUTER

   # API Keys (add at least one):
   OPENROUTER_API_KEY=your_key
   COHERE_API_KEY=your_key
   PREDIBASE_API_KEY=your_key
   ```

5. **Deploy**
   - Click "Create Web Service"
   - Render will build and deploy your app
   - You'll get a URL like `cluedo-arena.onrender.com`

6. **Auto-Deploy**
   - Render automatically deploys on every push to your branch
   - Check the "Logs" tab to monitor deployments

### Render Configuration File

The `render.yaml` file provides infrastructure-as-code configuration for Render.

---

## Vercel Deployment (Limited)

⚠️ **Warning**: Vercel's serverless architecture has limitations with Socket.IO. The application may not work correctly due to:
- WebSocket connections require persistent server instances
- Vercel functions are stateless and short-lived
- Socket.IO reconnection issues

**We recommend Railway or Render instead.**

However, if you still want to try Vercel:

### Deployment Steps

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Deploy**
   ```bash
   cd cluedo_game_engine
   vercel
   ```

3. **Set Environment Variables**
   ```bash
   vercel env add NODE_ENV production
   vercel env add LLM_BACKEND OPENROUTER
   vercel env add OPENROUTER_API_KEY
   # Add other API keys as needed
   ```

4. **Deploy to Production**
   ```bash
   vercel --prod
   ```

### Known Issues on Vercel
- WebSocket connections may drop frequently
- Games might disconnect mid-session
- Human player mode may be unreliable
- Better suited for static frontend hosting

### Alternative: Hybrid Deployment
- **Frontend**: Deploy on Vercel
- **Backend**: Deploy on Railway/Render
- Update Socket.IO client URL to point to your backend

---

## Environment Variables

All platforms require these environment variables:

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NODE_ENV` | Node environment | `production` |
| `PORT` | Server port | `8080` |
| `LLM_BACKEND` | LLM provider to use | `OPENROUTER`, `COHERE`, or `PREDIBASE` |

### LLM Provider API Keys (at least one required)

| Variable | Provider | Get API Key |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter | [openrouter.ai](https://openrouter.ai) |
| `COHERE_API_KEY` | Cohere | [cohere.com](https://cohere.com) |
| `PREDIBASE_API_KEY` | Predibase | [predibase.com](https://predibase.com) |

### Example `.env` File (for local development)

```bash
# Server Configuration
NODE_ENV=development
PORT=8080

# LLM Backend Selection
LLM_BACKEND=OPENROUTER

# API Keys (add at least one)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxx
COHERE_API_KEY=xxxxxxxxxxxxx
PREDIBASE_API_KEY=xxxxxxxxxxxxx

# Optional: Site Information
SITE_URL=http://localhost:8080
SITE_NAME=Cluedo Arena
```

---

## Local Testing

Before deploying, test the application locally:

1. **Install Dependencies**
   ```bash
   cd cluedo_game_engine
   npm install
   ```

2. **Set Up Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start Development Server**
   ```bash
   npm start
   ```

4. **Access the Application**
   - Open browser to `http://localhost:8080`
   - Test all three game modes:
     - 🎮 Play as Human
     - 👁️ Watch Single Game
     - 📊 Watch Multiple Games

5. **Verify WebSocket Connection**
   - Check for "Connected" status on the landing page
   - Monitor browser console for Socket.IO connection logs
   - Test reconnection by temporarily stopping the server

---

## Troubleshooting

### Issue: "Disconnected" status on landing page

**Solution**:
- Verify the server is running
- Check that PORT environment variable matches server configuration
- Ensure firewall allows WebSocket connections

### Issue: API errors during gameplay

**Solution**:
- Verify API keys are set correctly
- Check API key has sufficient credits
- Review server logs for specific error messages
- Try switching LLM_BACKEND to a different provider

### Issue: Games freeze or don't progress

**Solution**:
- Check server logs for LLM timeout errors
- Verify network connection is stable
- Ensure LLM provider is not rate-limiting requests
- Try refreshing the page and restarting the game

### Issue: Build fails on deployment

**Solution**:
- Verify Node.js version compatibility (recommended: v18+)
- Check `package.json` has all required dependencies
- Review build logs for specific error messages
- Ensure build command is `npm install`

---

## Post-Deployment Checklist

- [ ] Application deployed successfully
- [ ] Environment variables configured
- [ ] Can access the landing page
- [ ] Connection status shows "Connected"
- [ ] Tested "Play as Human" mode
- [ ] Tested "Watch Single Game" mode
- [ ] Tested "Watch Multiple Games" mode
- [ ] WebSocket connection remains stable
- [ ] No errors in server logs

---

## Support

If you encounter issues:

1. Check the server logs on your platform
2. Review the troubleshooting section above
3. Verify all environment variables are set correctly
4. Test locally first to isolate platform-specific issues

---

## Summary

**Quick Recommendation**:
1. ✅ **Use Railway** for the easiest deployment experience
2. ✅ **Use Render** as a great alternative
3. ❌ **Avoid Vercel** due to Socket.IO limitations

Both Railway and Render offer free tiers perfect for testing and hobby projects. For production use with guaranteed uptime, consider their paid plans.

Happy deploying! 🚀
