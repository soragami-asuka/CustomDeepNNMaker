//==================================
// 設定項目(列挙型) ※書き込み可能
//==================================
#ifndef __GRAVISBELL_I_NN_LAYER_CONFIG_EX_ENUM_H__
#define __GRAVISBELL_I_NN_LAYER_CONFIG_EX_ENUM_H__

#include"NNLayerInterface/ILayerConfig.h"

namespace Gravisbell {
namespace NeuralNetwork {

	class ILayerConfigItemEx_Enum : public ILayerConfigItem_Enum
	{
	public:
		/** コンストラクタ */
		ILayerConfigItemEx_Enum(){}
		/** デストラクタ */
		virtual ~ILayerConfigItemEx_Enum(){}


	public:
		/** 列挙値を追加する.
			同一IDが既に存在する場合失敗する.
			@param szEnumID		追加するID
			@param szEnumName	追加する名前
			@param szComment	追加するコメント分.
			@return 成功した場合、追加されたアイテムの番号が返る. 失敗した場合は負の値が返る. */
		virtual S32 AddEnumItem(const wchar_t szEnumID[], const wchar_t szEnumName[], const wchar_t szComment[]) = 0;

		/** 列挙値を削除する.
			@param num	削除する列挙番号
			@return 成功した場合0 */
		virtual S32 EraseEnumItem(S32 num) = 0;
		/** 列挙値を削除する
			@param szEnumID 削除する列挙ID
			@return 成功した場合0 */
		virtual S32 EraseEnumItem(const wchar_t szEnumID[]) = 0;

		/** デフォルト値を設定する.	番号指定.
			@param num デフォルト値に設定する番号. 
			@return 成功した場合0 */
		virtual S32 SetDefaultItem(S32 num) = 0;
		/** デフォルト値を設定する.	ID指定. 
			@param szID デフォルト値に設定するID. 
			@return 成功した場合0 */
		virtual S32 SetDefaultItem(const wchar_t szEnumID[]) = 0;

	};

}	// NeuralNetwork
}	// Gravisbell

#endif // __I_NN_LAYER_CONFIG_EX_ENUM_H__