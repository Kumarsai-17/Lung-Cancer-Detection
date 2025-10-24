// Mobile Navigation
document.addEventListener('DOMContentLoaded', function() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', function() {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
        
        // Close menu when clicking on a link
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                hamburger.classList.remove('active');
                navMenu.classList.remove('active');
            });
        });
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add scroll effect to navbar
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 100) {
                navbar.style.background = 'rgba(255, 255, 255, 0.98)';
                navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
            } else {
                navbar.style.background = 'rgba(255, 255, 255, 0.95)';
                navbar.style.boxShadow = 'none';
            }
        });
    }
    
    // Animate elements on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    document.querySelectorAll('.feature-card, .option-card, .tech-card, .recommendation-card').forEach(el => {
        observer.observe(el);
    });
    
    // Form validation and enhancement
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                const originalText = submitBtn.innerHTML;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                submitBtn.disabled = true;
                
                // Re-enable button after 30 seconds as fallback
                setTimeout(() => {
                    submitBtn.innerHTML = originalText;
                    submitBtn.disabled = false;
                }, 30000);
            }
        });
    });
    
    // Enhanced checkbox interactions
    document.querySelectorAll('.checkbox-item input[type="checkbox"]').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const label = this.nextElementSibling;
            if (this.checked) {
                label.style.transform = 'scale(1.02)';
                setTimeout(() => {
                    label.style.transform = 'scale(1)';
                }, 150);
            }
        });
    });
    
    // Progress indicator for forms
    const featureForm = document.getElementById('features-form');
    if (featureForm) {
        const checkboxes = featureForm.querySelectorAll('input[type="checkbox"]');
        const requiredFields = featureForm.querySelectorAll('input[required], select[required]');
        
        function updateProgress() {
            let completed = 0;
            let total = requiredFields.length;
            
            requiredFields.forEach(field => {
                if (field.value.trim() !== '') {
                    completed++;
                }
            });
            
            const progress = (completed / total) * 100;
            
            // Create or update progress bar
            let progressBar = document.querySelector('.form-progress');
            if (!progressBar) {
                progressBar = document.createElement('div');
                progressBar.className = 'form-progress';
                progressBar.innerHTML = `
                    <div class="progress-bar">
                        <div class="progress-fill"></div>
                    </div>
                    <span class="progress-text">Form completion: <span class="progress-percentage">0%</span></span>
                `;
                featureForm.insertBefore(progressBar, featureForm.firstChild);
            }
            
            const progressFill = progressBar.querySelector('.progress-fill');
            const progressText = progressBar.querySelector('.progress-percentage');
            
            progressFill.style.width = progress + '%';
            progressText.textContent = Math.round(progress) + '%';
        }
        
        requiredFields.forEach(field => {
            field.addEventListener('input', updateProgress);
            field.addEventListener('change', updateProgress);
        });
        
        updateProgress();
    }
    
    // Tooltip functionality
    function createTooltip(element, text) {
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.textContent = text;
        document.body.appendChild(tooltip);
        
        element.addEventListener('mouseenter', function(e) {
            tooltip.style.display = 'block';
            tooltip.style.left = e.pageX + 10 + 'px';
            tooltip.style.top = e.pageY - 30 + 'px';
        });
        
        element.addEventListener('mouseleave', function() {
            tooltip.style.display = 'none';
        });
        
        element.addEventListener('mousemove', function(e) {
            tooltip.style.left = e.pageX + 10 + 'px';
            tooltip.style.top = e.pageY - 30 + 'px';
        });
    }
    
    // Add tooltips to feature labels
    document.querySelectorAll('.checkbox-item label').forEach(label => {
        const text = label.querySelector('.label-text').textContent;
        const tooltips = {
            'Air Pollution Exposure': 'Regular exposure to polluted air in urban or industrial areas',
            'Dust Allergy': 'Allergic reactions to dust particles',
            'Occupational Hazards': 'Workplace exposure to harmful substances',
            'Genetic Risk': 'Family history of lung cancer or related conditions',
            'Chronic Lung Disease': 'Pre-existing lung conditions like COPD or asthma',
            'Chest Pain': 'Persistent or recurring chest discomfort',
            'Coughing Blood': 'Presence of blood in cough or sputum',
            'Clubbing of Finger Nails': 'Enlarged fingertips and curved nails'
        };
        
        if (tooltips[text]) {
            createTooltip(label, tooltips[text]);
        }
    });
    
    // Auto-save form data
    const autoSaveForm = document.getElementById('features-form');
    if (autoSaveForm) {
        const formData = {};
        
        // Load saved data
        const savedData = localStorage.getItem('lungcare-form-data');
        if (savedData) {
            try {
                const parsed = JSON.parse(savedData);
                Object.keys(parsed).forEach(key => {
                    const field = autoSaveForm.querySelector(`[name="${key}"]`);
                    if (field) {
                        if (field.type === 'checkbox') {
                            field.checked = parsed[key];
                        } else {
                            field.value = parsed[key];
                        }
                    }
                });
            } catch (e) {
                console.log('Error loading saved form data');
            }
        }
        
        // Save data on change
        autoSaveForm.addEventListener('input', function(e) {
            const field = e.target;
            if (field.type === 'checkbox') {
                formData[field.name] = field.checked;
            } else {
                formData[field.name] = field.value;
            }
            
            localStorage.setItem('lungcare-form-data', JSON.stringify(formData));
        });
        
        // Clear saved data on successful submission
        autoSaveForm.addEventListener('submit', function() {
            localStorage.removeItem('lungcare-form-data');
        });
    }
    
    // Enhanced image preview with additional info
    const imageInput = document.getElementById('image-input');
    if (imageInput) {
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Validate file type
                const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
                if (!validTypes.includes(file.type)) {
                    showNotification('Please select a valid image file (JPEG, PNG, GIF)', 'error');
                    return;
                }
                
                // Validate file size (16MB)
                if (file.size > 16 * 1024 * 1024) {
                    showNotification('File size must be less than 16MB', 'error');
                    return;
                }
                
                showNotification('Image uploaded successfully!', 'success');
            }
        });
    }
    
    // Notification system
    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
                <span>${message}</span>
            </div>
            <button class="notification-close">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
        
        // Manual close
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.remove();
        });
        
        // Animate in
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
    }
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit forms
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const activeForm = document.querySelector('form:focus-within');
            if (activeForm) {
                const submitBtn = activeForm.querySelector('button[type="submit"]');
                if (submitBtn && !submitBtn.disabled) {
                    submitBtn.click();
                }
            }
        }
        
        // Escape to close modals or go back
        if (e.key === 'Escape') {
            const backBtn = document.querySelector('.btn-secondary');
            if (backBtn && backBtn.textContent.includes('Back')) {
                backBtn.click();
            }
        }
    });
    
    // Performance monitoring
    if ('performance' in window) {
        window.addEventListener('load', function() {
            setTimeout(() => {
                const perfData = performance.getEntriesByType('navigation')[0];
                const loadTime = perfData.loadEventEnd - perfData.loadEventStart;
                
                if (loadTime > 3000) {
                    console.log('Page load time is high:', loadTime + 'ms');
                }
            }, 0);
        });
    }
    
    // Accessibility enhancements
    document.querySelectorAll('button, a, input, select').forEach(element => {
        element.addEventListener('focus', function() {
            this.style.outline = '2px solid #3b82f6';
            this.style.outlineOffset = '2px';
        });
        
        element.addEventListener('blur', function() {
            this.style.outline = 'none';
        });
    });
    
    // Dark mode toggle (if needed in future)
    function toggleDarkMode() {
        document.body.classList.toggle('dark-mode');
        localStorage.setItem('dark-mode', document.body.classList.contains('dark-mode'));
    }
    
    // Load dark mode preference
    if (localStorage.getItem('dark-mode') === 'true') {
        document.body.classList.add('dark-mode');
    }
});

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Add CSS for notifications and progress bar
const additionalCSS = `
.notification {
    position: fixed;
    top: 100px;
    right: 20px;
    background: white;
    border-radius: 0.75rem;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    z-index: 1001;
    transform: translateX(400px);
    transition: transform 0.3s ease;
    border-left: 4px solid #3b82f6;
    max-width: 400px;
}

.notification.show {
    transform: translateX(0);
}

.notification-success {
    border-left-color: #10b981;
}

.notification-error {
    border-left-color: #ef4444;
}

.notification-content {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex: 1;
}

.notification-content i {
    font-size: 1.25rem;
}

.notification-success i {
    color: #10b981;
}

.notification-error i {
    color: #ef4444;
}

.notification-close {
    background: none;
    border: none;
    cursor: pointer;
    color: #6b7280;
    padding: 0.25rem;
    border-radius: 0.25rem;
    transition: all 0.2s ease;
}

.notification-close:hover {
    background: #f3f4f6;
    color: #374151;
}

.form-progress {
    background: white;
    padding: 1rem;
    border-radius: 0.75rem;
    margin-bottom: 2rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    border: 1px solid #e5e7eb;
}

.progress-bar {
    height: 0.5rem;
    background: #e5e7eb;
    border-radius: 0.25rem;
    overflow: hidden;
    margin-bottom: 0.5rem;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    border-radius: 0.25rem;
    transition: width 0.3s ease;
}

.progress-text {
    font-size: 0.875rem;
    color: #6b7280;
    font-weight: 500;
}

.progress-percentage {
    color: #1e3a8a;
    font-weight: 600;
}

.tooltip {
    position: absolute;
    background: #1f2937;
    color: white;
    padding: 0.5rem 0.75rem;
    border-radius: 0.375rem;
    font-size: 0.875rem;
    z-index: 1002;
    display: none;
    max-width: 200px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    pointer-events: none;
}

.tooltip::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 5px solid transparent;
    border-top-color: #1f2937;
}

@media (max-width: 768px) {
    .notification {
        right: 10px;
        left: 10px;
        max-width: none;
        transform: translateY(-100px);
    }
    
    .notification.show {
        transform: translateY(0);
    }
}
`;

// Inject additional CSS
const style = document.createElement('style');
style.textContent = additionalCSS;
document.head.appendChild(style);