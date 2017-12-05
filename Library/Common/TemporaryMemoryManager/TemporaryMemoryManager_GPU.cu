// TemporaryMemoryManager.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"

#include"Library/Common/TemporaryMemoryManager.h"

#include<map>

#include<thrust/device_vector.h>


namespace Gravisbell {
namespace Common {

	class TemporaryMemoryManager_GPU : public ITemporaryMemoryManager
	{
	private:
		std::map<GUID, std::map<std::wstring, U32>>	lpBufferSize;	/**< バッファのサイズ一覧 */
		std::map<std::wstring, thrust::device_vector<BYTE>>	lpBuffer;		/**< バッファ本体 */

	public:
		/** コンストラクタ */
		TemporaryMemoryManager_GPU()
			:	ITemporaryMemoryManager()
		{
		}

		/** デストラクタ */
		virtual ~TemporaryMemoryManager_GPU()
		{
		}

	public:
		/** バッファサイズを登録する.
			@param	i_layerGUID		レイヤーのGUID.
			@param	i_szCode		使用方法を定義したID.
			@param	i_bufferSize	バッファのサイズ. バイト単位. */
		ErrorCode SetBufferSize(GUID i_layerGUID, const wchar_t i_szCode[], U32 i_bufferSize)
		{
			this->lpBufferSize[i_layerGUID][(std::wstring)i_szCode] = i_bufferSize;

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** バッファサイズを取得する.
			@param	i_layerGUID		レイヤーのGUID.
			@param	i_szCode		使用方法を定義したID. */
		U32 GetBufferSize(GUID i_layerGUID, const wchar_t i_szCode[])const
		{
			auto it_guid = this->lpBufferSize.find(i_layerGUID);
			if(it_guid == this->lpBufferSize.end())
				return 0;

			auto it_code = it_guid->second.find(i_szCode);
			if(it_code == it_guid->second.end())
				return 0;

			return it_code->second;
		}

		/** バッファを取得する */
		BYTE* GetBufer(GUID i_layerGUID, const wchar_t i_szCode[])
		{
			U32 bufferSize = lpBufferSize[i_layerGUID][i_szCode];
			thrust::device_vector<BYTE>& buffer = this->lpBuffer[i_szCode];

			if(buffer.size() < bufferSize)
				buffer.resize(bufferSize);

			return thrust::raw_pointer_cast(&buffer[0]);
		}
	};


	TemporaryMemoryManager_API ITemporaryMemoryManager* CreateTemporaryMemoryManagerGPU()
	{
		return new TemporaryMemoryManager_GPU();
	}

}	// Common
}	// Gravisbell
