-- APA-style figures for the PDF: caption above the image, and the "*Note.*" paragraph that follows a figure in the
-- Markdown kept inside the same float, so LaTeX can move the figure without separating it from its note.
-- Used by build_pdf.sh and build_pdf.bat (pandoc --lua-filter=apa_figures.lua).

local function is_figure(b)
  return b.t == "Para" and #b.content == 1 and b.content[1].t == "Image"
    and (b.content[1].title or ""):sub(1, 4) == "fig:"
end

local function is_note(b)
  return b and b.t == "Para" and b.content[1] and b.content[1].t == "Emph"
    and pandoc.utils.stringify(b.content[1]) == "Note."
end

local function latex_escape(s)
  return (s:gsub("([&%%$#_{}])", "\\%1"))
end

function Pandoc(doc)
  local out, blocks, i = {}, doc.blocks, 1
  while i <= #blocks do
    local b = blocks[i]
    if is_figure(b) then
      local img = b.content[1]
      img.title = ""  -- a plain image inside our own figure environment
      table.insert(out, pandoc.RawBlock("latex", "\\begin{figure}[htbp]\\centering"))
      table.insert(out, pandoc.RawBlock("latex", "\\caption{" .. latex_escape(pandoc.utils.stringify(img.caption)) .. "}"))
      img.caption = {}
      table.insert(out, pandoc.Plain({img}))
      if is_note(blocks[i + 1]) then
        table.insert(out, pandoc.RawBlock("latex", "\\par\\smallskip\\raggedright"))
        table.insert(out, blocks[i + 1])
        i = i + 1
      end
      table.insert(out, pandoc.RawBlock("latex", "\\end{figure}"))
    else
      table.insert(out, b)
    end
    i = i + 1
  end
  doc.blocks = out
  return doc
end
