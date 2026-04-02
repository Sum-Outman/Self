import React, { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import {
  Video,
  VideoOff,
  Mic,
  MicOff,
  MessageSquare,
  Settings,
  UserPlus,
  Users,
  Monitor,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Loader2,
  Copy,
} from 'lucide-react';
import toast from 'react-hot-toast';

// WebRTC连接状态
type WebRTCState = 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error';

// 组件模式（未来扩展）
type VideoChatMode = 'peer' | 'agi';

// 组件属性（未来扩展）
interface VideoChatProps {
  mode?: VideoChatMode;
  agiModel?: string;
}

// 视频聊天会话接口
interface VideoSession {
  id: string;
  title: string;
  participants: string[];
  createdAt: Date;
  active: boolean;
}

// 视频聊天消息接口
interface VideoMessage {
  id: string;
  content: string;
  sender: string;
  timestamp: Date;
  type: 'text' | 'system';
}

const VideoChat: React.FC<VideoChatProps> = ({ mode: _mode = 'peer', agiModel: _agiModel = 'default' }) => {
  // 本地视频流引用
  const webcamRef = useRef<Webcam>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  
  // 本地状态
  const [isVideoEnabled, setIsVideoEnabled] = useState(true);
  const [isAudioEnabled, setIsAudioEnabled] = useState(true);
  const [isScreenSharing, setIsScreenSharing] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [webrtcState, setWebrtcState] = useState<WebRTCState>('idle');
  
  // 连接信息
  const [roomId, setRoomId] = useState('');
  const [peerConnection, setPeerConnection] = useState<RTCPeerConnection | null>(null);
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [dataChannel, setDataChannel] = useState<RTCDataChannel | null>(null);
  const [, setSignalingWebSocket] = useState<WebSocket | null>(null);
  
  // 聊天消息
  const [messages, setMessages] = useState<VideoMessage[]>([
    {
      id: '1',
      content: '视频聊天已就绪，请加入或创建房间',
      sender: 'system',
      timestamp: new Date(),
      type: 'system'
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  
  // 会话列表
  const [sessions, _setSessions] = useState<VideoSession[]>([
    {
      id: 'default-session',
      title: '默认视频会话',
      participants: ['用户A', '用户B'],
      createdAt: new Date(),
      active: true
    }
  ]);
  
  // 初始化本地媒体流
  const initializeLocalStream = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: isVideoEnabled ? {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        } : false,
        audio: isAudioEnabled ? {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } : false
      });
      
      setLocalStream(stream);
      
      // 设置本地视频预览
      if (webcamRef.current && webcamRef.current.video) {
        webcamRef.current.video.srcObject = stream;
      }
      
      toast.success('摄像头和麦克风已启动');
      return stream;
    } catch (error) {
      console.error('获取媒体设备失败:', error);
      toast.error('无法访问摄像头或麦克风');
      return null;
    }
  }, [isVideoEnabled, isAudioEnabled]);
  
  // 初始化WebRTC连接
  const initializeWebRTC = useCallback(() => {
    if (!localStream) return null;
    
    const configuration = {
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
        { urls: 'stun:stun2.l.google.com:19302' },
        { urls: 'stun:stun3.l.google.com:19302' },
        { urls: 'stun:stun4.l.google.com:19302' }
      ]
    };
    
    const pc = new RTCPeerConnection(configuration);
    
    // 添加本地流到连接
    localStream.getTracks().forEach(track => {
      pc.addTrack(track, localStream);
    });
    
    // 监听远程流
    pc.ontrack = (event) => {
      const [remoteStream] = event.streams;
      setRemoteStream(remoteStream);
      
      if (remoteVideoRef.current) {
        remoteVideoRef.current.srcObject = remoteStream;
      }
    };
    
    // 监听连接状态
    pc.onconnectionstatechange = () => {
      switch (pc.connectionState) {
        case 'connected':
          setWebrtcState('connected');
          toast.success('视频连接已建立');
          break;
        case 'disconnected':
          setWebrtcState('disconnected');
          toast.error('视频连接断开');
          break;
        case 'failed':
          setWebrtcState('error');
          toast.error('视频连接失败');
          break;
        case 'closed':
          setWebrtcState('idle');
          break;
      }
    };
    
    // 监听ICE连接状态
    pc.oniceconnectionstatechange = () => {
      console.log('ICE连接状态:', pc.iceConnectionState);
    };
    
    // 监听ICE候选项
    pc.onicecandidate = (event) => {
      if (event.candidate) {
        // 这里应该通过信令服务器发送ICE候选项给对端
        console.log('ICE候选项:', event.candidate);
      }
    };
    
    // 创建数据通道
    const dc = pc.createDataChannel('chat', {
      ordered: true,
      maxRetransmits: 3
    });
    
    dc.onopen = () => {
      console.log('数据通道已打开');
      setDataChannel(dc);
      toast.success('数据通道已连接');
    };
    
    dc.onclose = () => {
      console.log('数据通道已关闭');
      setDataChannel(null);
      toast.error('数据通道已断开');
    };
    
    dc.onerror = (error) => {
      console.error('数据通道错误:', error);
      toast.error('数据通道错误');
    };
    
    dc.onmessage = (event) => {
      try {
        const messageData = JSON.parse(event.data);
        const newMessage: VideoMessage = {
          id: Date.now().toString(),
          content: messageData.content,
          sender: messageData.sender || '远程用户',
          timestamp: new Date(),
          type: 'text'
        };
        setMessages(prev => [...prev, newMessage]);
      } catch (error) {
        console.error('解析消息失败:', error, event.data);
      }
    };
    
    // 监听远程数据通道
    pc.ondatachannel = (event) => {
      const remoteDc = event.channel;
      setDataChannel(remoteDc);
      
      remoteDc.onopen = () => {
        console.log('远程数据通道已打开');
        toast.success('数据通道已连接');
      };
      
      remoteDc.onmessage = (event) => {
        try {
          const messageData = JSON.parse(event.data);
          const newMessage: VideoMessage = {
            id: Date.now().toString(),
            content: messageData.content,
            sender: messageData.sender || '远程用户',
            timestamp: new Date(),
            type: 'text'
          };
          setMessages(prev => [...prev, newMessage]);
        } catch (error) {
          console.error('解析远程消息失败:', error, event.data);
        }
      };
    };
    
    setPeerConnection(pc);
    return { pc, dc };
  }, [localStream]);
  
  // 创建或加入房间
  const handleJoinRoom = useCallback(async (roomId?: string) => {
    try {
      setWebrtcState('connecting');
      toast.loading('正在连接视频房间...', { id: 'connecting' });
      
      // 初始化本地流
      const stream = await initializeLocalStream();
      if (!stream) {
        toast.dismiss('connecting');
        setWebrtcState('error');
        return;
      }
      
      // 初始化WebRTC连接
      const webrtcResult = initializeWebRTC();
      if (!webrtcResult) {
        toast.dismiss('connecting');
        setWebrtcState('error');
        return;
      }
      
      const { pc } = webrtcResult;
      
      // 创建offer
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      
      // 通过信令服务器发送offer - 完整实现
      // 建立WebSocket连接到信令服务器
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.host}/api/ws/chat`;
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        // 发送offer到信令服务器
        ws.send(JSON.stringify({
          type: 'webrtc_signaling',
          roomId: roomId || `room-${Date.now().toString(36)}`,
          action: 'offer',
          sdp: offer,
          timestamp: new Date().toISOString()
        }));
        setSignalingWebSocket(ws);
        console.log('WebSocket信令连接已建立，offer已发送');
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'webrtc_signaling' && data.action === 'answer') {
            // 收到answer，设置远程描述
            pc.setRemoteDescription(new RTCSessionDescription(data.sdp))
              .then(() => console.log('远程描述设置成功'))
              .catch(err => console.error('设置远程描述失败:', err));
          } else if (data.type === 'webrtc_signaling' && data.action === 'ice_candidate') {
            // 收到ICE candidate
            pc.addIceCandidate(new RTCIceCandidate(data.candidate))
              .then(() => console.log('ICE candidate添加成功'))
              .catch(err => console.error('添加ICE candidate失败:', err));
          }
        } catch (error) {
          console.error('处理信令消息失败:', error);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket信令错误:', error);
        toast.error('信令服务器连接失败');
      };
      
      ws.onclose = () => {
        console.log('WebSocket信令连接关闭');
        setSignalingWebSocket(null);
      };
      
      // 生成或使用传入的房间ID
      const newRoomId = roomId || `room-${Date.now().toString(36)}`;
      setRoomId(newRoomId);
      
      toast.dismiss('connecting');
      
      // 添加系统消息
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        content: `已加入房间: ${newRoomId}`,
        sender: 'system',
        timestamp: new Date(),
        type: 'system'
      }]);
      
      toast.success(`已加入房间: ${newRoomId}`);
      
    } catch (error) {
      console.error('加入房间失败:', error);
      toast.dismiss('connecting');
      toast.error('加入房间失败');
      setWebrtcState('error');
    }
  }, [initializeLocalStream, initializeWebRTC]);
  
  // 离开房间
  const handleLeaveRoom = useCallback(() => {
    if (dataChannel) {
      dataChannel.close();
      setDataChannel(null);
    }
    
    if (peerConnection) {
      peerConnection.close();
      setPeerConnection(null);
    }
    
    if (localStream) {
      localStream.getTracks().forEach(track => track.stop());
      setLocalStream(null);
    }
    
    setRemoteStream(null);
    setWebrtcState('idle');
    
    // 添加系统消息
    setMessages(prev => [...prev, {
      id: Date.now().toString(),
      content: '已离开视频房间',
      sender: 'system',
      timestamp: new Date(),
      type: 'system'
    }]);
    
    toast.success('已离开视频房间');
  }, [peerConnection, localStream, dataChannel]);
  
  // 切换视频
  const toggleVideo = useCallback(async () => {
    const newVideoState = !isVideoEnabled;
    setIsVideoEnabled(newVideoState);
    
    if (localStream) {
      const videoTrack = localStream.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.enabled = newVideoState;
      }
    }
    
    toast.success(`视频 ${newVideoState ? '已开启' : '已关闭'}`);
  }, [isVideoEnabled, localStream]);
  
  // 切换音频
  const toggleAudio = useCallback(() => {
    const newAudioState = !isAudioEnabled;
    setIsAudioEnabled(newAudioState);
    
    if (localStream) {
      const audioTrack = localStream.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.enabled = newAudioState;
      }
    }
    
    toast.success(`音频 ${newAudioState ? '已开启' : '已关闭'}`);
  }, [isAudioEnabled, localStream]);
  
  // 开始/停止屏幕共享
  const toggleScreenShare = useCallback(async () => {
    try {
      if (!isScreenSharing) {
        // 开始屏幕共享
        const screenStream = await navigator.mediaDevices.getDisplayMedia({
          video: {
            width: { ideal: 1920 },
            height: { ideal: 1080 },
            frameRate: { ideal: 30 }
          },
          audio: true
        });
        
        if (peerConnection) {
          // 移除旧的视频轨道
          const senders = peerConnection.getSenders();
          const videoSender = senders.find(sender => 
            sender.track?.kind === 'video'
          );
          
          if (videoSender) {
            const screenTrack = screenStream.getVideoTracks()[0];
            videoSender.replaceTrack(screenTrack);
          }
          
          // 监听屏幕共享停止
          screenStream.getVideoTracks()[0].onended = () => {
            setIsScreenSharing(false);
            toast('屏幕共享已停止');
          };
        }
        
        setIsScreenSharing(true);
        toast.success('已开始屏幕共享');
      } else {
        // 停止屏幕共享，恢复摄像头
        if (peerConnection && localStream) {
          const senders = peerConnection.getSenders();
          const videoSender = senders.find(sender => 
            sender.track?.kind === 'video'
          );
          
          if (videoSender && localStream) {
            const cameraTrack = localStream.getVideoTracks()[0];
            videoSender.replaceTrack(cameraTrack);
          }
        }
        
        setIsScreenSharing(false);
        toast.success('已停止屏幕共享');
      }
    } catch (error) {
      console.error('屏幕共享失败:', error);
      toast.error('屏幕共享失败');
    }
  }, [isScreenSharing, peerConnection, localStream]);

  // 开始/停止录制
  const toggleRecording = useCallback(async () => {
    try {
      if (!isRecording) {
        // 开始录制
        const stream = localStream || await initializeLocalStream();
        if (!stream) {
          toast.error('无法访问媒体设备');
          return;
        }
        
        const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus') 
          ? 'video/webm;codecs=vp9,opus'
          : MediaRecorder.isTypeSupported('video/webm;codecs=vp8,opus')
          ? 'video/webm;codecs=vp8,opus'
          : 'video/webm';
        
        const recorder = new MediaRecorder(stream, {
          mimeType,
          videoBitsPerSecond: 2500000, // 2.5 Mbps
          audioBitsPerSecond: 128000   // 128 kbps
        });
        
        recorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            recordedChunksRef.current.push(event.data);
          }
        };
        
        recorder.onstop = () => {
          const blob = new Blob(recordedChunksRef.current, { type: mimeType });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `video-chat-${new Date().toISOString().replace(/[:.]/g, '-')}.webm`;
          a.click();
          URL.revokeObjectURL(url);
          recordedChunksRef.current = [];
        };
        
        recorder.start(1000); // 收集1秒的数据块
        recorderRef.current = recorder;
        setIsRecording(true);
        toast.success('录制已开始');
      } else {
        // 停止录制
        if (recorderRef.current && recorderRef.current.state !== 'inactive') {
          recorderRef.current.stop();
        }
        setIsRecording(false);
        toast.success('录制已停止，文件正在下载中...');
      }
    } catch (error) {
      console.error('录制失败:', error);
      toast.error('录制失败');
      setIsRecording(false);
    }
  }, [isRecording, localStream, initializeLocalStream]);
  
  // 发送聊天消息
  const handleSendMessage = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    
    if (!inputMessage.trim()) {
      toast.error('请输入消息内容');
      return;
    }
    
    const newMessage: VideoMessage = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: '用户',
      timestamp: new Date(),
      type: 'text'
    };
    
    setMessages(prev => [...prev, newMessage]);
    setInputMessage('');
    
    // 通过数据通道发送消息给对端
    if (dataChannel && dataChannel.readyState === 'open') {
      try {
        const messageData = {
          content: inputMessage,
          sender: '用户',
          timestamp: new Date().toISOString()
        };
        dataChannel.send(JSON.stringify(messageData));
        console.log('消息已通过数据通道发送:', inputMessage);
      } catch (error) {
        console.error('通过数据通道发送消息失败:', error);
        toast.error('消息发送失败');
      }
    } else {
      console.log('数据通道不可用，消息仅在本地显示');
      toast.error('数据通道不可用，消息仅在本地显示');
    }
  }, [inputMessage, dataChannel]);
  
  // 复制房间ID
  const copyRoomId = useCallback(() => {
    if (roomId) {
      navigator.clipboard.writeText(roomId);
      toast.success('房间ID已复制到剪贴板');
    } else {
      toast.error('未加入房间');
    }
  }, [roomId]);
  
  // 组件挂载时初始化
  useEffect(() => {
    // 检查浏览器支持
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast.error('浏览器不支持WebRTC');
      setWebrtcState('error');
      return;
    }
    
    return () => {
      // 组件卸载时清理
      handleLeaveRoom();
      
      // 停止录制
      if (recorderRef.current && recorderRef.current.state !== 'inactive') {
        recorderRef.current.stop();
      }
    };
  }, [handleLeaveRoom]);
  
  // 格式化时间
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
  };
  
  return (
    <div className="flex h-full bg-gray-50 dark:bg-gray-900">
      {/* 左侧会话列表 */}
      <div className="w-64 border-r border-gray-200 dark:border-gray-700 flex flex-col">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            视频会话
          </h2>
          <div className="space-y-2">
            <button
              onClick={() => handleJoinRoom()}
              disabled={webrtcState === 'connecting' || webrtcState === 'connected'}
              className="w-full flex items-center justify-center px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-800 to-gray-700 rounded-lg hover:from-gray-900 hover:to-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {webrtcState === 'connecting' ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  连接中...
                </>
              ) : webrtcState === 'connected' ? (
                <>
                  <CheckCircle className="w-4 h-4 mr-2" />
                  已连接
                </>
              ) : (
                <>
                  <Video className="w-4 h-4 mr-2" />
                  创建/加入房间
                </>
              )}
            </button>
            
            {webrtcState === 'connected' && (
              <button
                onClick={handleLeaveRoom}
                className="w-full flex items-center justify-center px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-900 to-gray-700 rounded-lg hover:from-gray-900 hover:to-gray-800"
              >
                <VideoOff className="w-4 h-4 mr-2" />
                离开房间
              </button>
            )}
          </div>
        </div>
        
        <div className="flex-1 overflow-auto p-2">
          {sessions.map(session => (
            <div
              key={session.id}
              className={`p-3 rounded-lg mb-2 cursor-pointer transition-colors ${
                session.active
                  ? 'bg-gray-700 dark:bg-gray-900/30 border border-gray-500 dark:border-gray-900'
                  : 'hover:bg-gray-100 dark:hover:bg-gray-800'
              }`}
            >
              <div className="flex items-center mb-2">
                <Video className={`w-4 h-4 mr-2 ${
                  session.active ? 'text-gray-800 dark:text-gray-400' : 'text-gray-500'
                }`} />
                <h4 className="font-medium text-gray-900 dark:text-white truncate">
                  {session.title}
                </h4>
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                <div className="flex items-center">
                  <Users className="w-3 h-3 mr-1" />
                  {session.participants.length} 人参与
                </div>
                <div className="mt-1">
                  创建时间: {session.createdAt.toLocaleDateString('zh-CN')}
                </div>
              </div>
            </div>
          ))}
        </div>
        
        {/* 房间信息 */}
        {webrtcState === 'connected' && (
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-gray-700 dark:text-gray-300 mb-1">
                  房间ID
                </label>
                <div className="flex items-center space-x-2">
                  <code className="flex-1 text-sm bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">
                    {roomId}
                  </code>
                  <button
                    onClick={copyRoomId}
                    className="p-1 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                    title="复制房间ID"
                  >
                    <Copy className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 主视频区域 */}
      <div className="flex-1 flex flex-col">
        {/* 视频区域头部 */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <div className="flex items-center">
            <div className="w-10 h-10 bg-gradient-to-r from-gray-700 to-gray-600 rounded-full flex items-center justify-center mr-3">
              <Video className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="font-semibold text-gray-900 dark:text-white">
                Self AGI 视频聊天
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {webrtcState === 'connected' 
                  ? '已连接' 
                  : webrtcState === 'connecting' 
                    ? '连接中...' 
                    : '未连接'}
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            {/* 连接状态指示器 */}
            <div className={`flex items-center px-3 py-1 rounded-full text-sm ${
              webrtcState === 'connected'
                ? 'bg-gray-600 dark:bg-gray-900/30 text-gray-800 dark:text-gray-400'
                : webrtcState === 'connecting'
                  ? 'bg-gray-700 dark:bg-gray-900/30 text-gray-900 dark:text-gray-400'
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
            }`}>
              {webrtcState === 'connected' ? (
                <>
                  <CheckCircle className="w-4 h-4 mr-1" />
                  已连接
                </>
              ) : webrtcState === 'connecting' ? (
                <>
                  <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                  连接中
                </>
              ) : (
                <>
                  <AlertCircle className="w-4 h-4 mr-1" />
                  未连接
                </>
              )}
            </div>
            
            <button className="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800" title="设置">
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* 视频和聊天区域 */}
        <div className="flex-1 flex p-4 space-x-4">
          {/* 视频区域 */}
          <div className="flex-1 flex flex-col">
            <div className="flex-1 flex space-x-4">
              {/* 本地视频 */}
              <div className="flex-1 bg-gray-900 rounded-lg overflow-hidden">
                {localStream ? (
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    videoConstraints={{
                      width: { ideal: 1280 },
                      height: { ideal: 720 },
                      frameRate: { ideal: 30 }
                    }}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-800 to-gray-900">
                    <div className="text-center">
                      <VideoOff className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                      <p className="text-gray-400">本地视频预览</p>
                      <p className="text-sm text-gray-500 mt-2">
                        {webrtcState === 'connected' ? '视频已连接' : '等待连接...'}
                      </p>
                    </div>
                  </div>
                )}
                
                {/* 本地视频控制标签 */}
                <div className="absolute bottom-4 left-4 bg-black/50 backdrop-blur-sm rounded-lg px-3 py-1 text-white text-sm">
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-gray-600 rounded-full mr-2" />
                    本地视频
                  </div>
                </div>
              </div>

              {/* 远程视频 */}
              <div className="flex-1 bg-gray-900 rounded-lg overflow-hidden">
                {remoteStream ? (
                  <video
                    ref={remoteVideoRef}
                    autoPlay
                    playsInline
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-800 to-gray-900">
                    <div className="text-center">
                      <UserPlus className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                      <p className="text-gray-400">远程视频</p>
                      <p className="text-sm text-gray-500 mt-2">
                        等待其他用户加入...
                      </p>
                    </div>
                  </div>
                )}
                
                {/* 远程视频标签 */}
                <div className="absolute bottom-4 right-4 bg-black/50 backdrop-blur-sm rounded-lg px-3 py-1 text-white text-sm">
                  <div className="flex items-center">
                    <div className={`w-2 h-2 rounded-full mr-2 ${
                      webrtcState === 'connected' ? 'bg-gray-600' : 'bg-gray-800'
                    }`} />
                    远程视频
                  </div>
                </div>
              </div>
            </div>

            {/* 视频控制栏 */}
            <div className="mt-4 flex items-center justify-center space-x-4">
              <button
                onClick={toggleVideo}
                className={`p-3 rounded-full ${
                  isVideoEnabled
                    ? 'bg-gradient-to-r from-gray-800 to-gray-700 text-white'
                    : 'bg-gray-200 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
                }`}
                title={isVideoEnabled ? '关闭视频' : '开启视频'}
              >
                {isVideoEnabled ? (
                  <Video className="w-5 h-5" />
                ) : (
                  <VideoOff className="w-5 h-5" />
                )}
              </button>
              
              <button
                onClick={toggleAudio}
                className={`p-3 rounded-full ${
                  isAudioEnabled
                    ? 'bg-gradient-to-r from-gray-800 to-gray-700 text-white'
                    : 'bg-gray-200 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
                }`}
                title={isAudioEnabled ? '静音' : '取消静音'}
              >
                {isAudioEnabled ? (
                  <Mic className="w-5 h-5" />
                ) : (
                  <MicOff className="w-5 h-5" />
                )}
              </button>
              
              <button
                onClick={toggleScreenShare}
                className={`p-3 rounded-full ${
                  isScreenSharing
                    ? 'bg-gradient-to-r from-orange-600 to-gray-900 text-white'
                    : 'bg-gray-200 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
                }`}
                title={isScreenSharing ? '停止屏幕共享' : '开始屏幕共享'}
              >
                <Monitor className="w-5 h-5" />
              </button>
              
              <button
                onClick={toggleRecording}
                className={`p-3 rounded-full ${
                  isRecording
                    ? 'bg-gradient-to-r from-gray-900 to-gray-700 text-white animate-pulse'
                    : 'bg-gray-200 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
                }`}
                title={isRecording ? '停止录制' : '开始录制'}
              >
                {isRecording ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <RefreshCw className="w-5 h-5" />
                )}
              </button>
              
              <button
                onClick={handleLeaveRoom}
                className="p-3 rounded-full bg-gradient-to-r from-gray-900 to-gray-700 text-white"
                title="挂断"
              >
                <VideoOff className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* 聊天区域 */}
          <div className="w-80 flex flex-col">
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                聊天消息
              </h3>
            </div>
            
            <div className="flex-1 overflow-auto bg-gray-100 dark:bg-gray-800 rounded-lg p-4 mb-4">
              {messages.map(message => (
                <div
                  key={message.id}
                  className={`mb-3 ${
                    message.type === 'system' ? 'text-center' : ''
                  }`}
                >
                  {message.type === 'system' ? (
                    <div className="text-xs text-gray-500 dark:text-gray-400 py-1">
                      {message.content}
                    </div>
                  ) : (
                    <div className={`max-w-full p-2 rounded-lg ${
                      message.sender === '用户'
                        ? 'bg-gradient-to-r from-gray-800 to-gray-700 text-white ml-auto'
                        : 'bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-white'
                    }`}>
                      <div className="text-xs opacity-80 mb-1">
                        {message.sender} · {formatTime(message.timestamp)}
                      </div>
                      <div className="text-sm">{message.content}</div>
                    </div>
                  )}
                </div>
              ))}
            </div>
            
            {/* 消息输入框 */}
            <form onSubmit={handleSendMessage} className="flex space-x-2">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="输入聊天消息..."
                className="flex-1 px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-1 focus:ring-gray-700 focus:border-gray-700 text-gray-900 dark:text-white"
                disabled={webrtcState !== 'connected'}
              />
              <button
                type="submit"
                disabled={!inputMessage.trim() || webrtcState !== 'connected'}
                className="px-4 py-2 bg-gradient-to-r from-gray-800 to-gray-700 text-white rounded-lg hover:from-gray-900 hover:to-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <MessageSquare className="w-5 h-5" />
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoChat;