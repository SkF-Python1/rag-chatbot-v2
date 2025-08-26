/**
 * RAG Chatbot Frontend Application
 */

class RAGChatbot {
    constructor() {
        this.apiBase = window.location.origin + '/api';
        this.conversationId = this.generateUUID();
        this.isLoading = false;
        
        this.initializeElements();
        this.attachEventListeners();
        this.loadStats();
        this.checkSystemHealth();
        
        // Auto-resize textarea
        this.setupAutoResize();
        
        console.log('RAG Chatbot initialized');
    }
    
    initializeElements() {
        // Chat elements
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.typingIndicator = document.getElementById('typingIndicator');
        
        // Upload elements
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        
        // Stats elements
        this.docCount = document.getElementById('docCount');
        this.chunkCount = document.getElementById('chunkCount');
        this.systemStatus = document.getElementById('systemStatus');
        
        // Notification element
        this.notification = document.getElementById('notification');
    }
    
    attachEventListeners() {
        // Chat functionality
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // File upload functionality
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e.target.files));
        
        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });
        
        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });
        
        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            this.handleFileUpload(e.dataTransfer.files);
        });
    }
    
    setupAutoResize() {
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
        });
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isLoading) return;
        
        // Add user message to chat
        this.addMessage('user', message);
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        
        // Show typing indicator
        this.showTyping();
        this.setLoading(true);
        
        try {
            const response = await fetch(`${this.apiBase}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    conversation_id: this.conversationId,
                    use_context: true
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Hide typing indicator
            this.hideTyping();
            
            // Add assistant response
            this.addMessage('assistant', data.response, data.sources);
            
            // Log performance
            if (data.response_time > 1.2) {
                console.warn(`Response time ${data.response_time.toFixed(3)}s exceeded target`);
            } else {
                console.log(`Response time: ${data.response_time.toFixed(3)}s`);
            }
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.hideTyping();
            this.addMessage('assistant', 'Sorry, I encountered an error while processing your request. Please try again.');
            this.showNotification('Error sending message. Please try again.', 'error');
        } finally {
            this.setLoading(false);
        }
    }
    
    addMessage(role, content, sources = []) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? '👤' : '🤖';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = content;
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        
        // Add sources if available
        if (sources && sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'message-sources';
            sourcesDiv.innerHTML = '<strong>Sources:</strong> ';
            
            sources.forEach((source, index) => {
                const sourceSpan = document.createElement('span');
                sourceSpan.className = 'source-item';
                sourceSpan.textContent = `Doc ${index + 1} (${(source.score * 100).toFixed(0)}%)`;
                sourceSpan.title = source.content;
                sourcesDiv.appendChild(sourceSpan);
            });
            
            messageContent.appendChild(sourcesDiv);
        }
        
        // Remove welcome message if it exists
        const welcomeMessage = this.chatMessages.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    showTyping() {
        this.typingIndicator.style.display = 'flex';
        this.scrollToBottom();
    }
    
    hideTyping() {
        this.typingIndicator.style.display = 'none';
    }
    
    setLoading(loading) {
        this.isLoading = loading;
        this.sendButton.disabled = loading;
        this.messageInput.disabled = loading;
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }
    
    async handleFileUpload(files) {
        if (!files || files.length === 0) return;
        
        this.showNotification('Uploading documents...', 'info');
        
        for (const file of files) {
            try {
                await this.uploadFile(file);
            } catch (error) {
                console.error('Error uploading file:', error);
                this.showNotification(`Error uploading ${file.name}`, 'error');
            }
        }
        
        // Refresh stats after upload
        setTimeout(() => this.loadStats(), 2000);
    }
    
    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${this.apiBase}/documents/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        this.showNotification(`${file.name} uploaded successfully!`, 'success');
        
        return data;
    }
    
    async loadStats() {
        try {
            const response = await fetch(`${this.apiBase}/documents/stats`);
            if (response.ok) {
                const stats = await response.json();
                this.updateStats(stats);
            }
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    }
    
    updateStats(stats) {
        this.docCount.textContent = Math.floor(stats.total_chunks / 10) || 0; // Rough estimate
        this.chunkCount.textContent = stats.total_vectors || 0;
    }
    
    async checkSystemHealth() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            if (response.ok) {
                this.systemStatus.textContent = 'Healthy';
                this.systemStatus.style.color = '#48bb78';
            } else {
                this.systemStatus.textContent = 'Warning';
                this.systemStatus.style.color = '#f6ad55';
            }
        } catch (error) {
            this.systemStatus.textContent = 'Offline';
            this.systemStatus.style.color = '#f56565';
            console.error('Health check failed:', error);
        }
        
        // Check again in 30 seconds
        setTimeout(() => this.checkSystemHealth(), 30000);
    }
    
    showNotification(message, type = 'success') {
        this.notification.textContent = message;
        this.notification.className = `notification ${type}`;
        this.notification.classList.add('show');
        
        setTimeout(() => {
            this.notification.classList.remove('show');
        }, 3000);
    }
    
    generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }
}

// Enhanced features for better UX
class ChatFeatures {
    constructor(chatbot) {
        this.chatbot = chatbot;
        this.initializeFeatures();
    }
    
    initializeFeatures() {
        this.addQuickActions();
        this.setupKeyboardShortcuts();
        this.addClearChatButton();
    }
    
    addQuickActions() {
        const quickActions = [
            "What documents do we have?",
            "Summarize the key points",
            "Find information about...",
            "Explain this topic"
        ];
        
        // Add quick action buttons when chat is empty
        const welcomeMessage = document.querySelector('.welcome-message');
        if (welcomeMessage) {
            const actionsDiv = document.createElement('div');
            actionsDiv.style.marginTop = '20px';
            
            quickActions.forEach(action => {
                const button = document.createElement('button');
                button.textContent = action;
                button.style.cssText = `
                    background: #4299e1;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    margin: 5px;
                    border-radius: 20px;
                    cursor: pointer;
                    font-size: 12px;
                    transition: background 0.3s ease;
                `;
                
                button.addEventListener('mouseenter', () => {
                    button.style.background = '#3182ce';
                });
                
                button.addEventListener('mouseleave', () => {
                    button.style.background = '#4299e1';
                });
                
                button.addEventListener('click', () => {
                    this.chatbot.messageInput.value = action;
                    this.chatbot.messageInput.focus();
                });
                
                actionsDiv.appendChild(button);
            });
            
            welcomeMessage.appendChild(actionsDiv);
        }
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + / to focus input
            if ((e.ctrlKey || e.metaKey) && e.key === '/') {
                e.preventDefault();
                this.chatbot.messageInput.focus();
            }
            
            // Escape to clear input
            if (e.key === 'Escape') {
                this.chatbot.messageInput.value = '';
                this.chatbot.messageInput.blur();
            }
        });
    }
    
    addClearChatButton() {
        const header = document.querySelector('.chat-header');
        const clearButton = document.createElement('button');
        clearButton.innerHTML = '🗑️';
        clearButton.title = 'Clear conversation';
        clearButton.style.cssText = `
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: none;
            border-radius: 50%;
            width: 32px;
            height: 32px;
            cursor: pointer;
            margin-left: 10px;
            transition: background 0.3s ease;
        `;
        
        clearButton.addEventListener('click', () => {
            this.clearChat();
        });
        
        clearButton.addEventListener('mouseenter', () => {
            clearButton.style.background = 'rgba(255, 255, 255, 0.3)';
        });
        
        clearButton.addEventListener('mouseleave', () => {
            clearButton.style.background = 'rgba(255, 255, 255, 0.2)';
        });
        
        header.appendChild(clearButton);
    }
    
    clearChat() {
        const messages = this.chatbot.chatMessages.querySelectorAll('.message');
        messages.forEach(message => message.remove());
        
        // Re-add welcome message
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'welcome-message';
        welcomeDiv.innerHTML = `
            <h3>👋 Welcome to your Internal Knowledge Assistant!</h3>
            <p>I can help you find information from your uploaded documents. Upload some documents to get started, then ask me anything!</p>
        `;
        
        this.chatbot.chatMessages.appendChild(welcomeDiv);
        
        // Generate new conversation ID
        this.chatbot.conversationId = this.chatbot.generateUUID();
        
        // Re-add quick actions
        this.addQuickActions();
        
        this.chatbot.showNotification('Chat cleared!', 'success');
    }
}

// Performance monitoring
class PerformanceMonitor {
    constructor() {
        this.metrics = {
            responseTime: [],
            uploadTime: [],
            errors: 0
        };
        
        this.startMonitoring();
    }
    
    startMonitoring() {
        // Monitor API performance
        this.interceptFetch();
        
        // Log performance metrics every minute
        setInterval(() => {
            this.logMetrics();
        }, 60000);
    }
    
    interceptFetch() {
        const originalFetch = window.fetch;
        
        window.fetch = async (...args) => {
            const startTime = performance.now();
            
            try {
                const response = await originalFetch(...args);
                const endTime = performance.now();
                const duration = endTime - startTime;
                
                const url = typeof args[0] === 'string' ? args[0] : args[0].url;
                
                if (url.includes('/chat')) {
                    this.metrics.responseTime.push(duration);
                } else if (url.includes('/upload')) {
                    this.metrics.uploadTime.push(duration);
                }
                
                return response;
            } catch (error) {
                this.metrics.errors++;
                throw error;
            }
        };
    }
    
    logMetrics() {
        if (this.metrics.responseTime.length > 0) {
            const avgResponseTime = this.metrics.responseTime.reduce((a, b) => a + b, 0) / this.metrics.responseTime.length;
            console.log(`Average response time: ${avgResponseTime.toFixed(2)}ms`);
            
            if (avgResponseTime > 1200) {
                console.warn('Response time exceeding target of 1.2s');
            }
        }
        
        if (this.metrics.errors > 0) {
            console.warn(`${this.metrics.errors} errors in the last minute`);
        }
        
        // Reset metrics
        this.metrics.responseTime = [];
        this.metrics.uploadTime = [];
        this.metrics.errors = 0;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const chatbot = new RAGChatbot();
    new ChatFeatures(chatbot);
    new PerformanceMonitor();
    
    // Add some helpful console messages
    console.log('🤖 RAG Chatbot Frontend Loaded');
    console.log('💡 Tip: Use Ctrl+/ to focus the input field');
    console.log('📊 Performance monitoring enabled');
}); 