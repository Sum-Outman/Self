import React from 'react';
import { Link } from 'react-router-dom';
import { Shield, ArrowLeft } from 'lucide-react';

const PrivacyPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* 头部 */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <div className="inline-flex items-center justify-center w-12 h-12 bg-gradient-to-br from-gray-600 to-gray-600 rounded-xl mr-4">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                  Self AGI 隐私政策
                </h1>
                <p className="text-gray-600 dark:text-gray-400 mt-1">
                  最后更新: 2026年3月11日
                </p>
              </div>
            </div>
            
            <Link
              to="/"
              className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              返回首页
            </Link>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <p className="text-gray-600 dark:text-gray-400">
              本隐私政策描述了 Self AGI 如何收集、使用、存储和保护您的个人信息。使用我们的服务即表示您同意本隐私政策中描述的做法。
            </p>
          </div>
        </div>

        {/* 主要内容 */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 border border-gray-200 dark:border-gray-700">
          <div className="prose prose-lg dark:prose-invert max-w-none">
            {/* 简介 */}
            <section className="mb-10">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                1. 引言
              </h2>
              <p className="text-gray-700 dark:text-gray-300 mb-4">
                在 Self AGI（以下简称"我们"、"我们的"或"本服务"），我们重视您的隐私。本隐私政策解释了当您使用我们的自主通用人工智能系统时，我们如何收集、使用、披露和保护您的个人信息。
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                本政策适用于我们通过网站、应用程序和服务的所有数据收集和使用实践。使用本服务即表示您同意按照本政策收集和使用信息。
              </p>
            </section>

            {/* 信息收集 */}
            <section className="mb-10">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                2. 我们收集的信息
              </h2>
              
              <div className="mb-6">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  2.1 您提供的信息
                </h3>
                <p className="text-gray-700 dark:text-gray-300 mb-2">
                  当您注册账户、使用服务或与我们联系时，我们可能收集：
                </p>
                <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
                  <li><strong>账户信息</strong>: 用户名、邮箱地址、密码、姓名</li>
                  <li><strong>个人资料信息</strong>: 头像、个人简介、联系信息</li>
                  <li><strong>通信内容</strong>: 与 Self AGI 的对话、查询、反馈</li>
                  <li><strong>支付信息</strong>: 账单地址、支付方式（如适用）</li>
                  <li><strong>上传内容</strong>: 文件、图像、音频、视频等多媒体内容</li>
                </ul>
              </div>
              
              <div className="mb-6">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  2.2 自动收集的信息
                </h3>
                <p className="text-gray-700 dark:text-gray-300 mb-2">
                  当您使用我们的服务时，我们可能自动收集：
                </p>
                <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
                  <li><strong>使用数据</strong>: IP地址、浏览器类型、设备信息、操作系统</li>
                  <li><strong>日志信息</strong>: 访问时间、页面浏览、点击流数据</li>
                  <li><strong>位置信息</strong>: 大致地理位置（基于IP地址）</li>
                  <li><strong>Cookies和类似技术</strong>: 用于功能、分析和偏好设置</li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  2.3 AGI系统交互数据
                </h3>
                <p className="text-gray-700 dark:text-gray-300 mb-2">
                  由于 Self AGI 是一个自主学习和演化的人工智能系统，我们可能收集：
                </p>
                <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
                  <li><strong>交互数据</strong>: 您与AGI系统的对话历史和模式</li>
                  <li><strong>学习数据</strong>: 系统从交互中学到的模式和知识</li>
                  <li><strong>性能数据</strong>: 系统响应的准确性和相关性指标</li>
                  <li><strong>演化数据</strong>: 系统能力随时间的变化和改进</li>
                </ul>
                <p className="text-gray-700 dark:text-gray-300 mt-2">
                  此类数据主要用于改进系统性能，并通常在匿名化和聚合后使用。
                </p>
              </div>
            </section>

            {/* 信息使用 */}
            <section className="mb-10">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                3. 我们如何使用您的信息
              </h2>
              
              <div className="mb-6">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  3.1 主要用途
                </h3>
                <p className="text-gray-700 dark:text-gray-300 mb-2">
                  我们使用收集的信息用于以下目的：
                </p>
                <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
                  <li>提供、维护和改进 Self AGI 服务</li>
                  <li>处理您的注册、交易和请求</li>
                  <li>与您沟通服务更新、安全警报和支持消息</li>
                  <li>个性化您的体验和提供相关内容</li>
                  <li>监控和分析服务使用和趋势</li>
                  <li>检测、预防和解决技术问题</li>
                  <li>训练和优化AGI系统性能</li>
                </ul>
              </div>
              
              <div className="mb-6">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  3.2 AGI系统训练
                </h3>
                <p className="text-gray-700 dark:text-gray-300">
                  Self AGI 是一个自主学习和演化的人工智能系统。您的交互数据可能用于：
                </p>
                <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300 mt-2">
                  <li>训练和改进AGI模型的能力和准确性</li>
                  <li>增强系统的理解和响应能力</li>
                  <li>促进系统的自主演化和适应</li>
                  <li>开发新功能和能力</li>
                </ul>
                <p className="text-gray-700 dark:text-gray-300 mt-2">
                  用于训练的数据通常在匿名化后使用，并与其他数据聚合以保护您的隐私。
                </p>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  3.3 法律依据
                </h3>
                <p className="text-gray-700 dark:text-gray-300 mb-2">
                  我们处理您的个人信息基于以下法律依据：
                </p>
                <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
                  <li><strong>履行合同</strong>: 为您提供所请求的服务</li>
                  <li><strong>合法利益</strong>: 改进服务、防止欺诈、确保安全</li>
                  <li><strong>同意</strong>: 您明确同意特定处理活动</li>
                  <li><strong>法律义务</strong>: 遵守适用的法律法规</li>
                </ul>
              </div>
            </section>

            {/* 信息共享 */}
            <section className="mb-10">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                4. 信息共享和披露
              </h2>
              
              <div className="mb-6">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  4.1 我们不会出售您的信息
                </h3>
                <p className="text-gray-700 dark:text-gray-300">
                  我们不会向第三方出售、交易或以其他方式转让您的个人信息，除非在以下情况下或得到您的明确同意。
                </p>
              </div>
              
              <div className="mb-6">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  4.2 可能的披露情况
                </h3>
                <p className="text-gray-700 dark:text-gray-300 mb-2">
                  我们可能在以下情况下共享您的信息：
                </p>
                <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
                  <li><strong>服务提供商</strong>: 帮助我们运营服务的可信第三方（如托管、支付处理）</li>
                  <li><strong>法律要求</strong>: 响应有效的法律程序或政府请求</li>
                  <li><strong>保护权利</strong>: 保护我们、我们的用户或公众的权利、财产或安全</li>
                  <li><strong>业务转移</strong>: 涉及合并、收购或资产出售的情况</li>
                  <li><strong>聚合数据</strong>: 与合作伙伴共享匿名、聚合的统计数据</li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  4.3 国际传输
                </h3>
                <p className="text-gray-700 dark:text-gray-300">
                  您的信息可能被传输到您所在国家以外的服务器进行处理。我们会采取适当措施确保您的信息在国际传输中得到充分保护。
                </p>
              </div>
            </section>

            {/* 数据安全 */}
            <section className="mb-10">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                5. 数据安全
              </h2>
              
              <div className="mb-6">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  5.1 安全措施
                </h3>
                <p className="text-gray-700 dark:text-gray-300 mb-2">
                  我们实施合理的技术和组织措施来保护您的个人信息，包括：
                </p>
                <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
                  <li>数据加密传输（SSL/TLS）</li>
                  <li>安全服务器基础设施和访问控制</li>
                  <li>定期安全审计和漏洞测试</li>
                  <li>员工隐私和安全培训</li>
                  <li>事件响应和灾难恢复计划</li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  5.2 安全责任共担
                </h3>
                <p className="text-gray-700 dark:text-gray-300">
                  虽然我们努力保护您的信息，但没有任何安全措施是100%安全的。您也有责任保护您的账户密码和访问凭据。请勿与他人共享您的登录信息。
                </p>
              </div>
            </section>

            {/* 数据保留 */}
            <section className="mb-10">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                6. 数据保留
              </h2>
              <p className="text-gray-700 dark:text-gray-300 mb-4">
                我们仅在实现本政策所述目的所需的时间内保留您的个人信息，除非法律要求或允许更长的保留期。
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                当不再需要您的信息时，我们会安全地删除或匿名化处理。某些信息可能因技术限制（如备份）或法律要求而保留更长时间。
              </p>
            </section>

            {/* 您的权利 */}
            <section className="mb-10">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                7. 您的隐私权利
              </h2>
              
              <div className="mb-6">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  7.1 权利列表
                </h3>
                <p className="text-gray-700 dark:text-gray-300 mb-2">
                  根据适用法律，您可能拥有以下权利：
                </p>
                <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
                  <li><strong>访问权</strong>: 请求访问我们持有的您的个人信息</li>
                  <li><strong>更正权</strong>: 请求更正不准确或不完整的个人信息</li>
                  <li><strong>删除权</strong>: 请求删除您的个人信息（"被遗忘权"）</li>
                  <li><strong>限制处理权</strong>: 请求限制处理您的个人信息</li>
                  <li><strong>数据可携权</strong>: 请求以结构化格式接收您的信息</li>
                  <li><strong>反对权</strong>: 反对基于合法利益的处理</li>
                  <li><strong>撤回同意权</strong>: 随时撤回您已给予的同意</li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  7.2 行使权利
                </h3>
                <p className="text-gray-700 dark:text-gray-300">
                  要行使您的隐私权利，请通过本政策末尾提供的联系方式与我们联系。我们将在法律要求的时间内响应您的请求。
                </p>
                <p className="text-gray-700 dark:text-gray-300 mt-2">
                  请注意，某些权利可能受到限制，例如当处理信息对履行合同或遵守法律义务是必要的时候。
                </p>
              </div>
            </section>

            {/* AGI特殊考虑 */}
            <section className="mb-10">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                8. AGI系统特殊考虑
              </h2>
              
              <div className="mb-6">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  8.1 自主学习和演化
                </h3>
                <p className="text-gray-700 dark:text-gray-300">
                  Self AGI 是一个自主学习和演化的系统。这意味着系统可能基于与您的交互进行学习和改进。虽然我们努力确保这种学习不损害您的隐私，但您应了解AGI系统的这一固有特性。
                </p>
              </div>
              
              <div className="mb-6">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  8.2 数据匿名化
                </h3>
                <p className="text-gray-700 dark:text-gray-300">
                  用于AGI系统训练的数据通常在匿名化后使用。然而，在某些情况下，系统可能保留某些交互模式用于个性化服务。您可以通过账户设置控制这种个性化的程度。
                </p>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  8.3 系统记忆
                </h3>
                <p className="text-gray-700 dark:text-gray-300">
                  Self AGI 具有长期和短期记忆能力，这可能影响系统与您的交互。您可以请求查看、更正或删除系统记忆中的特定信息，但请注意某些模式学习可能难以完全逆转。
                </p>
              </div>
            </section>

            {/* 儿童隐私 */}
            <section className="mb-10">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                9. 儿童隐私
              </h2>
              <p className="text-gray-700 dark:text-gray-300">
                我们的服务不针对13岁以下的儿童。我们不会故意收集13岁以下儿童的个人信息。如果您是父母或监护人并认为您的孩子向我们提供了个人信息，请与我们联系，我们将采取措施删除此类信息。
              </p>
            </section>

            {/* 政策变更 */}
            <section className="mb-10">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                10. 隐私政策变更
              </h2>
              <p className="text-gray-700 dark:text-gray-300 mb-4">
                我们可能不时更新本隐私政策。我们将通过在此页面上发布新政策来通知您任何更改。更改在此页面上发布后立即生效。
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                我们鼓励您定期查看本政策以了解变更。继续使用本服务即表示您接受修订后的隐私政策。
              </p>
            </section>

            {/* 联系我们 */}
            <section>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                11. 联系我们
              </h2>
              <p className="text-gray-700 dark:text-gray-300 mb-4">
                如果您对本隐私政策或我们的隐私实践有任何疑问、意见或关切，请通过以下方式与我们联系：
              </p>
              <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>邮箱:</strong> silencecrowtom@qq.com
                </p>
                <p className="text-gray-700 dark:text-gray-300 mt-2">
                  <strong>数据保护问题:</strong> 对于隐私相关问题，请在邮件主题中注明"隐私查询"
                </p>
                <p className="text-gray-700 dark:text-gray-300 mt-2">
                  <strong>最后更新:</strong> 2026年3月11日
                </p>
              </div>
            </section>
          </div>
        </div>

        {/* 底部导航 */}
        <div className="mt-8 flex justify-between items-center">
          <Link
            to="/terms"
            className="inline-flex items-center text-gray-600 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
          >
            ← 服务条款
          </Link>
          
          <div className="text-sm text-gray-500 dark:text-gray-400">
            <p>© {new Date().getFullYear()} Self AGI. 版权所有.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PrivacyPage;