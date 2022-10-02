## 个人网站

使用hugo+mainroad。

Step 1：本地预览
```
hugo server --verbose -D
```

Step 2：添加utteranc评论支持：

在模板footer处中添加以下代码后，静态编译生成到`./docs`目录下
```
<script src="https://utteranc.es/client.js"
    repo="zhaodaolimeng/my-github-pages"
    issue-term="pathname"
    theme="github-light"
    crossorigin="anonymous"
    async>
</script>
```

Step 3：编译静态网站：
```
hugo -D -d ./docs
```
