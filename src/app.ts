import { setupWechatPlatform } from '@/plugins/wechat_platform';
import { fetchFunc } from '@/plugins/fetch';

App({
    globalData: {},
    onLaunch() {
        // 展示本地存储能力
        const logs = wx.getStorageSync('logs') || []
        logs.unshift(Date.now())
        wx.setStorageSync('logs', logs)

        // 登录
        wx.login({
            success: _res => {
                // 发送 res.code 到后台换取 openId, sessionKey, unionId
                console.log("登录信息  res=", _res)
            }
        })

        setupWechatPlatform({
            fetchFunc: fetchFunc,
            canvas: wx.createOffscreenCanvas(0, 0)
        }, true);
    },
})
