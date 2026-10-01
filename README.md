<div align="center">
<strong>Boxuan Zhang の 个人博客</strong>
</div>

文章的 YAML 表头使用 `date` 记录发布日期，使用 `updated` 记录最后更新时间。新文章将 `updated` 设为与 `date` 相同的值，仅在后续修改旧文章时手动更新 `updated`，例如：

```yaml
date: "2026-09-27"
updated: "2026-10-01"
```

正文开头会以 Fluid 风格的浅蓝提示条显示 `Last updated on ...`。未填写 `updated` 的文章默认使用 `date`；构建不会自动修改更新时间。
