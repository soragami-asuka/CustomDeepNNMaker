//========================================
// パラメータ初期化ルーチン
//========================================
#ifndef __GRAVISBELL_I_NN_INITIALIZER_H__
#define __GRAVISBELL_I_NN_INITIALIZER_H__

#include"../../Common/Common.h"
#include"../../Common/ErrorCode.h"
#include"../../Common/Guiddef.h"
#include"../../Common/IODataStruct.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** 初期化ルーチン */
	class IInitializer
	{
	public:
		//===========================
		// コンストラクタ/デストラクタ
		//===========================
		/** コンストラクタ */
		IInitializer(){}
		/** デストラクタ */
		virtual ~IInitializer(){}

	public:
		//===========================
		// パラメータの値を取得
		//===========================

		/** パラメータの値を取得する.
			@param	i_inputCount	入力信号数.
			@param	i_outputCount	出力信号数. */
		virtual F32 GetParameter(U32 i_inputCount, U32 i_outputCount) = 0;
		/** パラメータの値を取得する.
			@param	i_inputStruct	入力構造.
			@param	i_outputStruct	出力構造. */
		virtual F32 GetParameter(const IODataStruct& i_inputStruct, const IODataStruct& i_outputStruct) = 0;
		/** パラメータの値を取得する.
			@param	i_inputStruct	入力構造.
			@param	i_inputCH		入力チャンネル数.
			@param	i_outputStruct	出力構造.
			@param	i_outputCH		出力チャンネル数. */
		virtual F32 GetParameter(const Vector3D<S32>& i_inputStruct, U32 i_inputCH, const Vector3D<S32>& i_outputStruct, U32 i_outputCH) = 0;

		/** パラメータの値を取得する.
			@param	i_inputCount		入力信号数.
			@param	i_outputCount		出力信号数.
			@param	i_parameterCount	設定するパラメータ数.
			@param	o_lpParameter		設定するパラメータの配列. */
		virtual ErrorCode GetParameter(U32 i_inputCount, U32 i_outputCount, U32 i_parameterCount, F32 o_lpParameter[]) = 0;
		/** パラメータの値を取得する.
			@param	i_inputStruct		入力構造.
			@param	i_outputStruct		出力構造.
			@param	i_parameterCount	設定するパラメータ数.
			@param	o_lpParameter		設定するパラメータの配列. */
		virtual ErrorCode GetParameter(const IODataStruct& i_inputStruct, const IODataStruct& i_outputStruct, U32 i_parameterCount, F32 o_lpParameter[]) = 0;
		/** パラメータの値を取得する.
			@param	i_inputStruct		入力構造.
			@param	i_inputCH			入力チャンネル数.
			@param	i_outputStruct		出力構造.
			@param	i_outputCH			出力チャンネル数.
			@param	i_parameterCount	設定するパラメータ数.
			@param	o_lpParameter		設定するパラメータの配列. */
		virtual ErrorCode GetParameter(const Vector3D<S32>& i_inputStruct, U32 i_inputCH, const Vector3D<S32>& i_outputStruct, U32 i_outputCH, U32 i_parameterCount, F32 o_lpParameter[]) = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell



#endif	__GRAVISBELL_I_NN_INITIALIZER_H__
