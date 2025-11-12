<template>
  <div class="welcome-page">
    <!-- Hero Section -->
    <section class="hero-section">
      <div class="hero-toolbar">
        <div class="logo">Phronesis</div>
        <div class="hero-nav">
          <button @click="$router.push('/login')" class="btn btn-secondary">Login</button>
          <button @click="$router.push('/signup')" class="btn btn-primary">Sign Up</button>
        </div>
      </div>

      <div class="hero-content">
        <div class="hero-text">
          <h1 class="hero-title">
            Welcome to <span class="highlight">Phronesis</span>
          </h1>
          <p class="hero-subtitle">
            Collaborative Model Training Platform for Sepsis Phenotyping
          </p>
          <p class="hero-description">
            Empowering data scientists and clinicians to work together in developing 
            and validating machine learning models for better patient outcomes.
          </p>
          <div class="hero-actions">
            <button @click="$router.push('/signup')" class="btn btn-primary btn-lg">
              Get Started
            </button>
            <button @click="scrollToFeatures" class="btn btn-secondary btn-lg">
              Learn More
            </button>
          </div>
        </div>
        <div class="hero-illustration">
          <div class="illustration-placeholder">
            <svg width="200" height="200" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3ZM19 19H5V5H19V19Z" fill="white"/>
              <path d="M13.5 13C14.88 13 16 11.88 16 10.5C16 9.12 14.88 8 13.5 8C12.12 8 11 9.12 11 10.5C11 11.88 12.12 13 13.5 13Z" fill="white"/>
              <path d="M8 16H19V14.83C19 13.25 16.67 12.25 14.5 12.25C12.33 12.25 10 13.25 10 14.83V16H8Z" fill="white"/>
            </svg>
          </div>
        </div>
      </div>
    </section>

    <!-- Features Section -->
    <section class="features-section" ref="featuresSection">
      <div class="container">
        <div class="section-header">
          <h2>Powerful Features</h2>
          <p>Everything you need for collaborative model development</p>
        </div>
        
        <div class="card-grid">
          <div class="card feature-card">
            <div class="feature-icon">📊</div>
            <h3>Train Models</h3>
            <p>
              Upload datasets and configure clustering algorithms including K-Means, 
              K-Means with DTW, and Latent Class Analysis.
            </p>
          </div>

          <div class="card feature-card">
            <div class="feature-icon">👥</div>
            <h3>Collaborate</h3>
            <p>
              Bridge the gap between data scientists and clinicians with 
              seamless feedback and annotation workflows.
            </p>
          </div>

          <div class="card feature-card">
            <div class="feature-icon">🔄</div>
            <h3>Compare Results</h3>
            <p>
              Side-by-side comparison of multiple model runs with 
              comprehensive visualizations and metrics.
            </p>
          </div>

          <div class="card feature-card">
            <div class="feature-icon">📈</div>
            <h3>Real-time Monitoring</h3>
            <p>
              Track model training progress in real-time with instant 
              notifications and status updates.
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- How It Works Section -->
    <section class="how-it-works-section">
      <div class="container">
        <div class="section-header">
          <h2>How It Works</h2>
          <p>Simple workflow for powerful results</p>
        </div>

        <div class="steps-container">
          <div class="step-card">
            <div class="step-number">1</div>
            <div class="step-content">
              <h3>Upload & Configure</h3>
              <p>
                Data scientists upload DuckDB datasets and configure model parameters 
                using YAML configuration files.
              </p>
            </div>
          </div>

          <div class="step-card">
            <div class="step-number">2</div>
            <div class="step-content">
              <h3>Train Models</h3>
              <p>
                Run clustering algorithms (K-Means, DTW, LCA) on your data with 
                automatic optimal cluster detection.
              </p>
            </div>
          </div>

          <div class="step-card">
            <div class="step-number">3</div>
            <div class="step-content">
              <h3>Clinical Review</h3>
              <p>
                Send results to clinicians for validation and feedback on the 
                discovered phenotypes.
              </p>
            </div>
          </div>

          <div class="step-card">
            <div class="step-number">4</div>
            <div class="step-content">
              <h3>Iterate & Improve</h3>
              <p>
                Incorporate clinical feedback and iterate on models to achieve 
                better patient stratification.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- CTA Section -->
    <section class="cta-section">
      <div class="cta-content">
        <h2>Ready to get started?</h2>
        <p>Join data scientists and clinicians in advancing sepsis research</p>
        <button @click="$router.push('/signup')" class="btn btn-primary btn-lg">
          Create Account
        </button>
      </div>
    </section>

    <!-- Footer -->
    <footer class="footer">
      <div class="container">
        <div class="footer-content">
          <div class="footer-left">
            <div class="logo">Phronesis</div>
            <p>Collaborative Model Training Platform</p>
            <p class="partnership">In partnership with SingHealth</p>
          </div>
          <div class="footer-right">
            <div class="footer-links">
              <a @click="$router.push('/login')">Login</a>
              <a @click="$router.push('/signup')">Sign Up</a>
            </div>
          </div>
        </div>
        <div class="footer-bottom">
          <p>&copy; 2025 Phronesis. All rights reserved.</p>
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const featuresSection = ref(null)

const scrollToFeatures = () => {
  featuresSection.value?.scrollIntoView({ behavior: 'smooth' })
}
</script>

<style scoped>
.welcome-page {
  min-height: 100vh;
  background: #f5f7fa;
}

/* Hero Section */
.hero-section {
  background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
  color: white;
  padding-bottom: 4rem;
}

.hero-toolbar {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  padding: 1rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.logo {
  font-size: 1.8rem;
  font-weight: 700;
  color: white;
  letter-spacing: -0.5px;
}

.hero-nav {
  display: flex;
  gap: 1rem;
}

.hero-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 4rem 2rem;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4rem;
  align-items: center;
}

.hero-title {
  font-size: 3.5rem;
  font-weight: 700;
  margin-bottom: 1rem;
  line-height: 1.2;
}

.highlight {
  color: #fff;
  text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
}

.hero-subtitle {
  font-size: 1.5rem;
  margin-bottom: 1rem;
  font-weight: 500;
  opacity: 0.95;
}

.hero-description {
  font-size: 1.1rem;
  margin-bottom: 2rem;
  opacity: 0.9;
  line-height: 1.6;
}

.hero-actions {
  display: flex;
  gap: 1rem;
}

.btn-lg {
  padding: 0.75rem 2rem;
  font-size: 1.1rem;
}

.hero-illustration {
  display: flex;
  justify-content: center;
  align-items: center;
}

.illustration-placeholder {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 20px;
  padding: 3rem;
  backdrop-filter: blur(10px);
  display: flex;
  justify-content: center;
  align-items: center;
}

/* Features Section */
.features-section {
  padding: 5rem 0;
}

.section-header {
  text-align: center;
  margin-bottom: 3rem;
}

.section-header h2 {
  font-size: 2.5rem;
  color: #2c3e50;
  margin-bottom: 0.5rem;
}

.section-header p {
  font-size: 1.2rem;
  color: #7f8c8d;
}

.feature-card {
  text-align: center;
  transition: transform 0.3s, box-shadow 0.3s;
  height: 100%;
}

.feature-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

.feature-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.feature-card h3 {
  font-size: 1.3rem;
  color: #2c3e50;
  margin-bottom: 0.75rem;
}

.feature-card p {
  color: #7f8c8d;
  line-height: 1.6;
}

/* How It Works Section */
.how-it-works-section {
  background: white;
  padding: 5rem 0;
}

.steps-container {
  max-width: 900px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.step-card {
  display: flex;
  gap: 2rem;
  align-items: start;
  padding: 2rem;
  border-left: 4px solid #3498db;
  background: #f8f9fa;
  border-radius: 8px;
  transition: transform 0.2s;
}

.step-card:hover {
  transform: translateX(8px);
}

.step-number {
  width: 60px;
  height: 60px;
  background: #3498db;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.8rem;
  font-weight: 700;
  flex-shrink: 0;
}

.step-content h3 {
  font-size: 1.5rem;
  color: #2c3e50;
  margin-bottom: 0.5rem;
}

.step-content p {
  color: #7f8c8d;
  line-height: 1.6;
}

/* CTA Section */
.cta-section {
  background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
  color: white;
  padding: 5rem 2rem;
  text-align: center;
}

.cta-content h2 {
  font-size: 2.5rem;
  margin-bottom: 1rem;
}

.cta-content p {
  font-size: 1.2rem;
  margin-bottom: 2rem;
  opacity: 0.9;
}

/* Footer */
.footer {
  background: #2c3e50;
  color: white;
  padding: 3rem 0 1rem 0;
}

.footer-content {
  display: flex;
  justify-content: space-between;
  margin-bottom: 2rem;
}

.footer-left .logo {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
}

.footer-left p {
  opacity: 0.8;
  margin: 0.25rem 0;
}

.partnership {
  font-style: italic;
  margin-top: 0.5rem;
}

.footer-links {
  display: flex;
  gap: 2rem;
}

.footer-links a {
  color: white;
  text-decoration: none;
  cursor: pointer;
  transition: opacity 0.2s;
}

.footer-links a:hover {
  opacity: 0.8;
}

.footer-bottom {
  padding-top: 1rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  text-align: center;
  opacity: 0.7;
}

/* Responsive Design */
@media (max-width: 768px) {
  .hero-content {
    grid-template-columns: 1fr;
    text-align: center;
  }

  .hero-title {
    font-size: 2.5rem;
  }

  .hero-actions {
    justify-content: center;
    flex-direction: column;
  }

  .hero-illustration {
    display: none;
  }

  .footer-content {
    flex-direction: column;
    gap: 2rem;
  }

  .step-card {
    flex-direction: column;
    text-align: center;
  }

  .step-number {
    margin: 0 auto;
  }
}
</style>