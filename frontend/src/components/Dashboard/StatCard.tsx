import React, { ReactNode } from 'react';

interface StatCardProps {
  title: string;
  value: string | number;
  change?: string;
  icon: ReactNode;
  color: 'blue' | 'green' | 'purple' | 'red' | 'yellow' | 'indigo' | 'gray';
}

const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  change,
  icon,
  color,
}) => {
  const colorClasses = {
    blue: {
      iconBg: 'bg-gray-600 dark:bg-gray-900',
      iconText: 'text-gray-800 dark:text-gray-400',
      changeText: 'text-gray-700 dark:text-gray-400',
    },
    green: {
      iconBg: 'bg-gray-600 dark:bg-gray-900',
      iconText: 'text-gray-700 dark:text-gray-400',
      changeText: 'text-gray-700 dark:text-gray-400',
    },
    purple: {
      iconBg: 'bg-gray-600 dark:bg-gray-900',
      iconText: 'text-gray-700 dark:text-gray-400',
      changeText: 'text-gray-700 dark:text-gray-400',
    },
    red: {
      iconBg: 'bg-gray-800 dark:bg-gray-900',
      iconText: 'text-gray-900 dark:text-gray-500',
      changeText: 'text-gray-700 dark:text-gray-400',
    },
    yellow: {
      iconBg: 'bg-gray-700 dark:bg-gray-900',
      iconText: 'text-gray-700 dark:text-gray-400',
      changeText: 'text-gray-700 dark:text-gray-400',
    },
    indigo: {
      iconBg: 'bg-gray-700 dark:bg-gray-900',
      iconText: 'text-gray-800 dark:text-gray-400',
      changeText: 'text-gray-700 dark:text-gray-400',
    },
    gray: {
      iconBg: 'bg-gray-600 dark:bg-gray-900',
      iconText: 'text-gray-800 dark:text-gray-400',
      changeText: 'text-gray-700 dark:text-gray-400',
    },
  };

  const colors = colorClasses[color];

  return (
    <div className="card p-6 hover:shadow-lg transition-shadow duration-200">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
            {title}
          </p>
          <div className="mt-2 flex items-baseline">
            <p className="text-3xl font-bold text-gray-900 dark:text-white">
              {value}
            </p>
            {change && (
              <span
                className={`ml-2 text-sm font-medium ${colors.changeText}`}
              >
                {change}
              </span>
            )}
          </div>
        </div>
        <div
          className={`${colors.iconBg} p-3 rounded-full`}
        >
          <div className={colors.iconText}>
            {icon}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatCard;