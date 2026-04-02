/**
 * 类名合并工具
 * 用于合并多个CSS类名，处理条件类名和对象类名
 * 
 * @example
 * cn('btn', 'btn-primary', isLoading && 'loading', { active: isActive })
 * // 输出: 'btn btn-primary loading active'
 */

type ClassValue = string | number | boolean | null | undefined | ClassArray | ClassObject;
interface ClassArray extends Array<ClassValue> {}
interface ClassObject extends Record<string, any> {}

export function cn(...inputs: ClassValue[]): string {
  const classes: string[] = [];
  
  for (const input of inputs) {
    if (!input) continue;
    
    if (typeof input === 'string' || typeof input === 'number') {
      classes.push(String(input));
    } else if (Array.isArray(input)) {
      classes.push(cn(...input));
    } else if (typeof input === 'object') {
      for (const [key, value] of Object.entries(input)) {
        if (value) {
          classes.push(key);
        }
      }
    }
  }
  
  return classes.filter(Boolean).join(' ');
}

export default cn;