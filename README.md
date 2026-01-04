# weapp-vite-tailwindcss-tdesign-template
`weapp-vite` 集成 `tailwindcss`, `tdesign` 的模板


## 项目创建
pnpm create weapp-vite@latest

## 使用方式

### 开发

- `pnpm dev`

- `pnpm dev --open` 可以打包并直接启动微信开发者工具

### 构建

`pnpm build`

### 打开微信开发者工具

`pnpm open`

### 生成组件/页面

`pnpm g path/to/your/component`

## 文档地址

0. `weapp-vite`: https://vite.icebreaker.top/
1. `weapp-tailwindcss`: https://tw.icebreaker.top/

## 问题记录：
1. 为什么 导出的tfjs(带nms)的模型， 进行推理后，会出现重复项； 如对一张图片(只有一个猫和一个狗)进行推理， 检测的结果是 1个狗和9个猫， 这9个猫的数据(包括置信度和检测宽的数值)一模一样

2. 导出的tfjs模型中，出现 Infinity
```
"quantization": {
    "dtype": "uint8",
    "min": -Infinity,
    "scale": 1.0,
    "original_dtype": "float32"
}
```

"tdesign-miniprogram": "^1.12.0",

"preloadRule": {
        "pages/index/index": {
          "network": "all",
          "packages": [
            "models_json",
            "models_bin1",
            "models_bin2",
            "models_bin3"
            ]
        }
    },