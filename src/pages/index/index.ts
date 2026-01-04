// index.ts

Page({
    data: {
      list: [
        {
          title: '器械清点',
          img: 'https://ai.flypot.cn/mp/ai-pocket/images/index-imagenet-bg.jpg',
          url: '/pages/ai/objectDetect/index'
        }
      ]
    },
  
    onShareAppMessage() {
      return {
        title: 'AI Pocket - 口袋里的 AI'
      }
    },
   
    handleCardClicked(e: any) {
      wx.navigateTo({
        url: e.currentTarget.dataset.url
      })
    },
  
    // goToCoupon: function () {
    //   wx.navigateToMiniProgram({
    //     "appId": "wxece3a9a4c82f58c9",
    //     "extraData": {},
    //     "path": "taoke/pages/shopping-guide/index?scene=9GWP2Ou"
    //   })
    // }
  });
  
  export {};
  