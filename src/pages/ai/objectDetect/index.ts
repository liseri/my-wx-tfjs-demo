// index.js
import { ObjectDetection } from '@/models/yolo11/model';

const { appWidth, appHeight, benchmarkLevel } = getApp().globalData

interface Detection {
    label: string;
    score: number;
    left: number;
    top: number;
    width: number;
    height: number;
}

interface ImageInfo {
    width: number;
    height: number;
}

Page({
    objectDetection: ObjectDetection,
    data: {
        objectDetection: null,
        imagePath: '',
        detections: [] as Detection[],
        devicePosition: 'back',
        flash: 'off',
        isWasmLoaded: false,
        runningMode: 'IMAGE',
        detectionStats: [] as {label: string, count: number}[]
    },

    onLoad() {
        if (typeof WXWebAssembly === 'undefined') {
            wx.showModal({
                title: '提示',
                content: '当前微信版本不支持 WebAssembly，请升级微信到最新版本',
                showCancel: false
            })
            return
        }
        // this.initTfjs()
    },

    onReady() {
      this.initClassifier()
    },

    initClassifier() {
      wx.showLoading({ title: '模型正在加载...' })
      this.objectDetection = new ObjectDetection({ width: appWidth, height: appHeight })
      this.objectDetection.load().then(() => {
        wx.hideLoading()
      }).catch(err => {
        console.log('模型加载报错：', err)
      })
    },

    async initTfjs() {
        wx.showLoading({
            title: '加载模型中...'
        })
        try {
            await model.load();
            this.setData({
                isWasmLoaded: true
            })
        } catch (error: any) {
            console.error('模型 初始化失败:', error)
            wx.hideLoading()
            wx.showModal({
                title: '错误',
                content: '模型加载失败: ' + error.message,
                showCancel: false
            })
        }

        wx.hideLoading()
        wx.showToast({
            title: '模型加载成功'
        })
    },

    // 拍照检测
    takePhoto() {
        if (!this.data.isWasmLoaded) {
            wx.showToast({
                title: '模型未加载完成',
                icon: 'none'
            })
            return
        }

        const ctx = wx.createCameraContext()
        ctx.takePhoto({
            quality: 'high',
            success: (res) => {
                this.setData({
                    imagePath: res.tempImagePath
                })
                this.detectObjects(res.tempImagePath)
            }
        })
    },

    // 选择图片
    chooseImage() {
        if (!this.data.isWasmLoaded) {
            wx.showToast({
                title: '模型未加载完成',
                icon: 'none'
            })
            return
        }

        wx.chooseImage({
            count: 1,
            sizeType: ['original', 'compressed'],
            sourceType: ['album', 'camera'],
            success: (res) => {
                const tempFilePaths = res.tempFilePaths
                this.setData({
                    imagePath: tempFilePaths[0]
                })
                this.detectObjects(tempFilePaths[0])
            }
        })
    },

    // 检测物体
    async detectObjects(imagePath: string) {
        wx.showLoading({
            title: '检测中...'
        })
        
        try {
            // 获取图片信息，用于计算显示尺寸
            const imageInfo = await this.getImageInfo(imagePath) as ImageInfo
            const imageWidth = imageInfo.width
            const imageHeight = imageInfo.height
            
            // 先将图片设置到data中，让页面渲染出来
            this.setData({
                imagePath: imagePath,
                detections: [] // 清空之前的检测结果
            })

            // 确保图片加载完成后再获取显示尺寸
            await new Promise<void>((resolve) => {
                // 图片加载需要时间，添加一个小延迟
                setTimeout(() => {
                    resolve()
                }, 100) // 100ms延迟，确保图片渲染完成
            })

            // 获取图片显示尺寸，用于模型检测
            const query = wx.createSelectorQuery()
            await new Promise<void>((resolve) => {
                query.select('#detectionImage').boundingClientRect()
                query.exec(async (res) => {
                    if (res && res[0]) {
                        let displayWidth = res[0].width
                        let displayHeight = res[0].height
                        console.log("图片显示尺寸：宽=" + displayWidth + ", 高=" + displayHeight)
                        
                        // 必须创建canvas，因为model.detect需要canvas输入
                        // 微信小程序的image组件是原生组件，不能直接传给tf.browser.fromPixels
                        const canvas = wx.createOffscreenCanvas({
                            type: '2d',
                            width: displayWidth,
                            height: displayHeight
                        })
                        const ctx = canvas.getContext('2d')
                        
                        // 设置Canvas尺寸
                        canvas.width = displayWidth
                        canvas.height = displayHeight
                        
                        // 将图片绘制到Canvas上
                        const img = canvas.createImage()
                        await new Promise<void>((imgResolve, imgReject) => {
                            img.onload = () => {
                                ctx.drawImage(img, 0, 0, displayWidth, displayHeight)
                                imgResolve()
                            }
                            img.onerror = imgReject
                            img.src = imagePath
                        })
                        
                        if (!this.objectDetection.isReady()) {
                             console.log("模型没有正确加载"); 
                             return;  
                        }
                        
                        // 调用model.detect，直接传递显示尺寸
                        const detectedObjects = await this.objectDetection.detect(canvas);
                        
                        console.log('检测结果:', detectedObjects);

                        // 处理检测结果
                        const processedDetections = this.processDetections(detectedObjects)
                        
                        // 计算检测统计信息
                        const detectionStats = this.calculateDetectionStats(processedDetections)
                        
                        this.setData({
                            detections: processedDetections,
                            detectionStats: detectionStats
                        })
                    }
                    resolve()
                })
            })

            wx.hideLoading()
        } catch (error: any) {
            console.error('检测失败:', error)
            wx.hideLoading()
            wx.showModal({
                title: '错误',
                content: '检测失败: ' + error.message,
                showCancel: false
            })
        }
    },

    // 获取图片信息
    getImageInfo(imagePath: string) {
        return new Promise<ImageInfo>((resolve, reject) => {
            wx.getImageInfo({
                src: imagePath,
                success: resolve,
                fail: reject
            })
        })
    },



    // 处理检测结果
    processDetections(result: any) {
        const processedDetections: Detection[] = []

        for (let detection of result.detections || result) {
            if (!detection.bbox) continue

            const label = detection.class
            const score = Math.round(detection.score * 100)

            const boundingBox = detection.bbox
            // 直接使用检测结果的坐标，因为model.detect已经根据displaySize调整了
            const left = boundingBox[0]
            const top = boundingBox[1]
            const width = boundingBox[2]
            const height = boundingBox[3]

            processedDetections.push({
                label,
                score,
                left,
                top,
                width,
                height
            })
        }

        return processedDetections
    },

    // 计算检测统计信息
    calculateDetectionStats(detections: Detection[]) {
        const statsMap = new Map<string, {count: number, totalScore: number}>()
        
        // 统计每个类别的数量和总得分
        for (const detection of detections) {
            if (statsMap.has(detection.label)) {
                const stat = statsMap.get(detection.label)!
                stat.count++
                stat.totalScore += detection.score
            } else {
                statsMap.set(detection.label, {
                    count: 1,
                    totalScore: detection.score
                })
            }
        }
        
        // 转换为数组并计算平均得分
        const statsArray = Array.from(statsMap.entries()).map(([label, data]) => ({
            label,
            count: data.count
        }))
        
        // 按平均置信度从高到低排序
        statsArray.sort((a, b) => b.count - a.count)
        
        return statsArray
    },

    // 摄像头错误回调
    error(e: any) {
        console.log('摄像头错误', e.detail)
    }
})