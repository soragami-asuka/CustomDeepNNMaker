//=======================================
// レイヤーDLLクラス
//=======================================
#ifndef __GRAVISBELL_I_TEMPORARY_MEMORY_MANAGER_H__
#define __GRAVISBELL_I_TEMPORARY_MEMORY_MANAGER_H__

#include"Guiddef.h"
#include"ErrorCode.h"

namespace Gravisbell {
namespace Common {

	class ITemporaryMemoryManager
	{
	public:
		/** コンストラクタ */
		ITemporaryMemoryManager(){}
		/** デストラクタ */
		virtual ~ITemporaryMemoryManager(){}

	public:
		/** バッファサイズを登録する.
			@param	i_layerGUID		レイヤーのGUID.
			@param	i_szCode		使用方法を定義したID.
			@param	i_bufferSize	バッファのサイズ. バイト単位. */
		virtual ErrorCode SetBufferSize(GUID i_layerGUID, const wchar_t i_szCode[], U32 i_bufferSize) = 0;

		/** バッファサイズを取得する.
			@param	i_layerGUID		レイヤーのGUID.
			@param	i_szCode		使用方法を定義したID.
			@param	i_bufferSize	バッファのサイズ. バイト単位. */
		virtual U32 GetBufferSize(GUID i_layerGUID, const wchar_t i_szCode[])const = 0;

		/** バッファを取得する */
//		virtual BYTE* GetBuffer(GUID i_layerGUID, const wchar_t i_szCode[]) = 0;

		/** バッファを予約して取得する */
		virtual BYTE* ReserveBuffer(GUID i_layerGUID, const wchar_t i_szCode[]) = 0;
		/** 予約済みバッファを開放する */
		virtual void RestoreBuffer(GUID i_layerGUID, const wchar_t i_szCode[]) = 0;
	};

}	// Layer
}	// Gravisbell

#endif