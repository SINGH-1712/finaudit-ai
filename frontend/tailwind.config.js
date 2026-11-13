// tailwind.config.js
export default {
    content: ['./index.html', './src/**/*.{js,jsx}'],
    theme: {
      extend: {
        colors: {
          brand: { 50:'#eef6ff',100:'#d9ebff',200:'#bcdcff',300:'#8ec6ff',400:'#59a5ff',500:'#2f83ff',600:'#1862f5',700:'#144ee1',800:'#173fb6',900:'#19398f' },
        },
      },
    },
    plugins: [],
  }