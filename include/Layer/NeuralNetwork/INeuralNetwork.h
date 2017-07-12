//=======================================
// ニューラルネットワーク本体定義
//=======================================
#ifndef __GRAVISBELL_I_NEURAL_NETWORK_H__
#define __GRAVISBELL_I_NEURAL_NETWORK_H__

#include"INNSingle2SingleLayer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class INeuralNetwork : public INNSingle2SingleLayer
	{
	public:
		/** コンストラクタ */
		INeuralNetwork(){}
		/** デストラクタ */
		virtual ~INeuralNetwork(){}

	public:
		//====================================
		// 学習設定
		//====================================
		/** 学習設定を取得する.
			@param	guid	取得対象レイヤーのGUID. */
		virtual const SettingData::Standard::IData* GetRuntimeParameter(const Gravisbell::GUID& guid)const = 0;

		/** 学習設定を設定する.
			int型、float型、enum型が対象.
			@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param) = 0;
		virtual ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, S32 i_param) = 0;
		/** 学習設定を設定する.
			int型、float型が対象.
			@param	guid		取得対象レイヤーのGUID. 指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param) = 0;
		virtual ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, F32 i_param) = 0;
		/** 学習設定を設定する.
			bool型が対象.
			@param	guid		取得対象レイヤーのGUID. 指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, bool i_param) = 0;
		virtual ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, bool i_param) = 0;
		/** 学習設定を設定する.
			string型が対象.
			@param	guid		取得対象レイヤーのGUID. 指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param) = 0;
		virtual ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, const wchar_t* i_param) = 0;

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif // __GRAVISBELL_I_NEURAL_NETWORK_H__
