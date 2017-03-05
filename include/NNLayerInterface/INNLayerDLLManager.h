//=======================================
// レイヤーDLLクラス
//=======================================
#ifndef __GRAVISBELL_I_NN_LAYER_DLL_MANAGER_H__
#define __GRAVISBELL_I_NN_LAYER_DLL_MANAGER_H__

#include"Common/Guiddef.h"
#include"Common/ErrorCode.h"

#include"INNLayerDLL.h"

namespace Gravisbell {
namespace NeuralNetwork {

	class ILayerDLLManager
	{
	public:
		/** コンストラクタ */
		ILayerDLLManager(){}
		/** デストラクタ */
		virtual ~ILayerDLLManager(){}

	public:
		/** DLLを読み込んで、管理に追加する.
			@param szFilePath		読み込むファイルのパス.
			@param o_addLayerCode	追加されたGUIDの格納先アドレス.
			@return	成功した場合0が返る. */
		virtual ErrorCode ReadLayerDLL(const wchar_t szFilePath[], GUID& o_addLayerCode) = 0;
		/** DLLを読み込んで、管理に追加する.
			@param szFilePath	読み込むファイルのパス.
			@return	成功した場合0が返る. */
		virtual ErrorCode ReadLayerDLL(const wchar_t szFilePath[]) = 0;

		/** 管理しているレイヤーDLLの数を取得する */
		virtual unsigned int GetLayerDLLCount()const = 0;
		/** 管理しているレイヤーDLLを番号指定で取得する.
			@param	num	取得するDLLの管理番号.
			@return 成功した場合はDLLクラスのアドレス. 失敗した場合はNULL */
		virtual const ILayerDLL* GetLayerDLLByNum(unsigned int num)const = 0;
		/** 管理しているレイヤーDLLをguid指定で取得する.
			@param guid	取得するDLLのGUID.
			@return 成功した場合はDLLクラスのアドレス. 失敗した場合はNULL */
		virtual const ILayerDLL* GetLayerDLLByGUID(GUID i_layerCode)const = 0;

		/** レイヤーDLLを削除する. */
		virtual ErrorCode EraseLayerDLL(GUID i_layerCode) = 0;
	};

}	// NeuralNetwork
}	// Gravisbell

#endif