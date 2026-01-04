export const getFrameSliceOptions = (frameWidth: number, frameHeight: number, displayWidth: number, displayHeight: number) => {
    let result = {
      start: [0, 0, 0],
      size: [-1, -1, 3]
    };
  
    const ratio = displayHeight / displayWidth;
  
    if (ratio > frameHeight / frameWidth) {
      // 计算宽度方向的切片
      const sliceWidth = Math.floor(frameHeight / ratio);
      result.start = [0, Math.max(0, Math.floor((frameWidth - sliceWidth) / 2)), 0];
      result.size = [-1, Math.min(sliceWidth, frameWidth), 3];
    } else {
      // 计算高度方向的切片
      const sliceHeight = Math.floor(ratio * frameWidth);
      result.start = [Math.max(0, Math.floor((frameHeight - sliceHeight) / 2)), 0, 0];
      result.size = [Math.min(sliceHeight, frameHeight), -1, 3];
    }
  
    return result;
  }