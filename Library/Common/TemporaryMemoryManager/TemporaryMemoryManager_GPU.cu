// TemporaryMemoryManager.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"

#include"Library/Common/TemporaryMemoryManager.h"

#include<map>
#include<list>

#include<thrust/device_vector.h>


namespace Gravisbell {
namespace Common {

	template<class BufferType>
	class TemporaryMemoryManager : public ITemporaryMemoryManager
	{
	private:
		struct BufferInfo
		{
			bool onReserved;						/**< 使用予約がされているか */
			GUID guid;								/**< 予約しているレイヤーのGUID */
			BufferType lpBuffer;	/**< バッファ本体 */

			BufferInfo()
				:	onReserved	(false)
				,	guid		()
				,	lpBuffer()
			{
			}
			BufferInfo(BufferInfo& info)
				:	onReserved	(info.onReserved)
				,	guid		(info.guid)
				,	lpBuffer	(info.lpBuffer)
			{
			}
			BufferInfo(bool i_onReserved, GUID i_guid, U32 i_bufferSize)
				:	onReserved	(i_onReserved)
				,	guid		(i_guid)
				,	lpBuffer	(i_bufferSize)
			{
			}
		};

		std::map<GUID, std::map<std::wstring, U32>>	lpBufferSize;				/**< バッファのサイズ一覧 */
		std::map<std::wstring, BufferType>	lpShareBuffer;		/**< 共有バッファ本体 */
		std::map<std::wstring, std::list<BufferInfo>>		lpReserveBuffer;	/**< 予約バッファ本体 */


	public:
		/** コンストラクタ */
		TemporaryMemoryManager()
			:	ITemporaryMemoryManager()
		{
		}

		/** デストラクタ */
		virtual ~TemporaryMemoryManager()
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
		BYTE* GetBuffer(GUID i_layerGUID, const wchar_t i_szCode[])
		{
			U32 bufferSize = lpBufferSize[i_layerGUID][i_szCode];
			thrust::device_vector<BYTE>& buffer = this->lpShareBuffer[i_szCode];

			if(buffer.size() < bufferSize)
				buffer.resize(bufferSize);

			return thrust::raw_pointer_cast(&buffer[0]);
		}

		/** バッファを予約して取得する */
		BYTE* ReserveBuffer(GUID i_layerGUID, const wchar_t i_szCode[])
		{
			// バッファサイズを取得
			U32 bufferSize = lpBufferSize[i_layerGUID][i_szCode];

			// 同一コードの空きバッファを検索
			auto& bufferList = this->lpReserveBuffer[i_szCode];
			auto it = bufferList.begin();
			while(it != bufferList.end())
			{
				if(!it->onReserved)
					break;
				if(it->guid == i_layerGUID)
					return thrust::raw_pointer_cast(&it->lpBuffer[0]);	// 予約済みで同一GUIDの場合終了

				it++;
			}

			if(it == bufferList.end())
			{
				// 空きがない場合は強制的に追加
				it = bufferList.insert(bufferList.end(), BufferInfo(true, i_layerGUID, bufferSize));
			}
			else
			{
				// バッファのサイズを確認して、必要なら拡張
				if(it->lpBuffer.size() < bufferSize)
					it->lpBuffer.resize(bufferSize);
			}

			it->onReserved = true;
			it->guid = i_layerGUID;

			return thrust::raw_pointer_cast(&it->lpBuffer[0]);
		}
		/** 予約済みバッファを開放する */
		void RestoreBuffer(GUID i_layerGUID, const wchar_t i_szCode[])
		{
			// 同一コードの予約済みバッファを検索
			auto& bufferList = this->lpReserveBuffer[i_szCode];
			auto it = bufferList.begin();
			while(it != bufferList.end())
			{
				if(it->guid == i_layerGUID)
				{
					it->onReserved = false;
					return;
				}

				it++;
			}
		}
	};


	TemporaryMemoryManager_API ITemporaryMemoryManager* CreateTemporaryMemoryManagerGPU()
	{
		return new TemporaryMemoryManager<thrust::device_vector<BYTE>>();
	}
	TemporaryMemoryManager_API ITemporaryMemoryManager* CreateTemporaryMemoryManagerCPU()
	{
		return new TemporaryMemoryManager<thrust::host_vector<BYTE>>();
	}

}	// Common
}	// Gravisbell
