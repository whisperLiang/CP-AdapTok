"""Seekable HTTP range reader for extracting a bounded subset of a large ZIP."""

import io
import urllib.request


class RangeReader(io.RawIOBase):
    def __init__(self, url, proxy=None, block_size=2**21):
        super().__init__()
        self.url, self.block_size, self.position = url, block_size, 0
        handler = urllib.request.ProxyHandler({"https": proxy}) if proxy else urllib.request.ProxyHandler()
        self.opener = urllib.request.build_opener(handler)
        with self.opener.open(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
            self.size = int(response.headers["Content-Length"])
        self.blocks, self.downloaded = {}, 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if self.position < 0:
            raise ValueError("negative archive position")
        return self.position

    def _block(self, index):
        if index not in self.blocks:
            start = index * self.block_size
            end = min(start + self.block_size, self.size) - 1
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            with self.opener.open(request, timeout=90) as response:
                if response.status != 206:
                    raise ValueError("archive server did not honor HTTP Range")
                data = response.read()
                if len(data) != end - start + 1:
                    raise ValueError("incomplete archive range")
            self.downloaded += len(data)
            self.blocks[index] = data
        return self.blocks[index]

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        result = bytearray()
        while size > 0:
            index, offset = divmod(self.position, self.block_size)
            block = self._block(index)
            count = min(size, len(block) - offset)
            result.extend(block[offset : offset + count])
            self.position += count
            size -= count
        return bytes(result)
