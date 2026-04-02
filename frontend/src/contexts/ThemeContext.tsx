import { createContext, useContext, useEffect, useState, ReactNode } from 'react';

type Theme = 'light' | 'dark' | 'auto';

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

interface ThemeProviderProps {
  children: ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  const [theme, setTheme] = useState<Theme>(() => {
    // 从localStorage获取保存的主题
    const savedTheme = localStorage.getItem('theme') as Theme;
    if (savedTheme === 'light' || savedTheme === 'dark' || savedTheme === 'auto') {
      return savedTheme;
    }
    
    // 默认为自动模式
    return 'auto';
  });

  useEffect(() => {
    const root = document.documentElement;
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    console.log(`主题切换: theme=${theme}, prefersDark=${prefersDark}, 添加dark类? ${theme === 'dark' || (theme === 'auto' && prefersDark)}`);
    
    if (theme === 'dark' || (theme === 'auto' && prefersDark)) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    
    // 保存到localStorage
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prevTheme => {
      if (prevTheme === 'light') return 'dark';
      if (prevTheme === 'dark') return 'auto';
      return 'light';
    });
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme必须在ThemeProvider内使用');
  }
  return context;
}