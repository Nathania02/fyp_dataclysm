<template>
  <footer class="footer">
    <div class="container">
      <div class="footer-content">
        <div class="footer-left">
          <div class="footer-logo">
            <span class="logo-icon">🧠</span>
            Phenosense
          </div>
          <p>Collaborative Model Training Platform</p>
          <p class="partnership">In partnership with SingHealth</p>
        </div>
        <div class="footer-right">
          <div class="footer-links">
            <router-link to="/about">About Us</router-link>
            <a @click="goToFeatures">Features</a>
            <router-link to="/login">Login</router-link>
            <router-link to="/signup">Sign Up</router-link>
          </div>
        </div>
      </div>
      
      <div class="footer-divider"></div>
      
      <div class="footer-disclaimer">
        <p class="disclaimer-text">
          ⚠️ <strong>Research Tool Notice:</strong> This platform is a purely a research tool developed 
          for academic purposes. All data is meant to be 
          stored securely on SingHealth infrastructure and handled in accordance with institutional 
          data protection policies.
        </p>
      </div>
      
      <div class="footer-bottom">
        <p>&copy; 2025 Dataclysm. All rights reserved.</p>
      </div>
    </div>
  </footer>
</template>

<script setup>
import { useRouter } from 'vue-router'

const router = useRouter()

const goToFeatures = () => {
  if (router.currentRoute.value.path === '/welcome') {
    const element = document.getElementById('features')
    if (element) {
      const navHeight = 70
      const elementPosition = element.getBoundingClientRect().top + window.pageYOffset
      const offsetPosition = elementPosition - navHeight
      
      window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth'
      })
    }
  } else {
    router.push('/welcome#features')
  }
}
</script>

<style scoped>
/* Footer */
.footer {
  background: #2c3e50;
  color: white;
  padding: 3rem 0 1rem 0;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
}

.footer-content {
  display: flex;
  justify-content: space-between;
  margin-bottom: 2rem;
}

.footer-logo {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.logo-icon {
  font-size: 1.5rem;
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
  align-items: start;
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

.footer-divider {
  height: 1px;
  background: rgba(255, 255, 255, 0.1);
  margin: 2rem 0;
}

.footer-disclaimer {
  margin-bottom: 2rem;
  padding: 1.5rem;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
  border-left: 4px solid #e74c3c;
}

.disclaimer-text {
  font-size: 0.9rem;
  line-height: 1.6;
  opacity: 0.9;
}

.footer-bottom {
  padding-top: 1rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  text-align: center;
  opacity: 0.7;
}

/* Responsive Design */
@media (max-width: 768px) {
  .footer-content {
    flex-direction: column;
    gap: 2rem;
  }
  
  .footer-links {
    flex-direction: column;
    gap: 1rem;
  }
}
</style>