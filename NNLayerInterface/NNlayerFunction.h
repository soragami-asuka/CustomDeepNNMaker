//=======================================
// ���C���[DLL�֐�
//=======================================
#ifndef __GRAVISBELL_NN_LAYER_FUNCTION_H__
#define __GRAVISBELL_NN_LAYER_FUNCTION_H__

#include"Common/Guiddef.h"
#include"Common/VersionCode.h"
#include"Common/ErrorCode.h"

#include"INNLayer.h"


namespace Gravisbell {
namespace NeuralNetwork {

	/** ���C���[���ʃR�[�h���擾����.
		@param o_layerCode	�i�[��o�b�t�@
		@return ���������ꍇ0 */
	typedef Gravisbell::ErrorCode (*FuncGetLayerCode)(GUID& o_layerCode);
	/** �o�[�W�����R�[�h���擾����.
		@param o_versionCode	�i�[��o�b�t�@
		@return ���������ꍇ0 */
	typedef Gravisbell::ErrorCode (*FuncGetVersionCode)(Gravisbell::VersionCode& o_versionCode);


	/** ���C���[�ݒ���쐬���� */
	typedef ILayerConfig* (*FuncCreateLayerConfig)(void);
	/** ���C���[�ݒ���쐬����
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@param o_useBufferSize ���ۂɓǂݍ��񂾃o�b�t�@�T�C�Y
		@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
	typedef ILayerConfig* (*FuncCreateLayerConfigFromBuffer)(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize);


	/** CPU�����p�̃��C���[���쐬 */
	typedef INNLayer* (*FuncCreateLayerCPU)(GUID guid);

	/** GPU�����p�̃��C���[���쐬 */
	typedef INNLayer* (*FuncCreateLayerGPU)(GUID guid);

}	// NeuralNetwork
}	// Gravisbell

#endif