//=======================================
// ���C���[DLL�N���X
//=======================================
#ifndef __GRAVISBELL_I_NN_LAYER_DLL_H__
#define __GRAVISBELL_I_NN_LAYER_DLL_H__

#include<guiddef.h>

#include"Common/VersionCode.h"

#include"INNLayer.h"

namespace Gravisbell {
namespace NeuralNetwork {

	class ILayerDLL
	{
	public:
		/** �R���X�g���N�^ */
		ILayerDLL(){}
		/** �f�X�g���N�^ */
		virtual ~ILayerDLL(){}

	public:
		/** ���C���[���ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual Gravisbell::ErrorCode GetLayerCode(GUID& o_layerCode)const = 0;
		/** �o�[�W�����R�[�h���擾����.
			@param o_versionCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual Gravisbell::ErrorCode GetVersionCode(Gravisbell::VersionCode& o_versionCode)const = 0;


		/** ���C���[�ݒ���쐬���� */
		virtual ILayerConfig* CreateLayerConfig(void)const = 0;
		/** ���C���[�ݒ���쐬����
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@param o_useBufferSize ���ۂɓǂݍ��񂾃o�b�t�@�T�C�Y
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		virtual ILayerConfig* CreateLayerConfigFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)const = 0;

		
		/** CPU�����p�̃��C���[���쐬.
			GUID�͎������蓖��. */
		virtual INNLayer* CreateLayerCPU()const = 0;
		/** CPU�����p�̃��C���[���쐬
			@param guid	�쐬���C���[��GUID */
		virtual INNLayer* CreateLayerCPU(GUID guid)const = 0;
		
		/** GPU�����p�̃��C���[���쐬.
			GUID�͎������蓖��. */
		virtual INNLayer* CreateLayerGPU()const = 0;
		/** GPU�����p�̃��C���[���쐬 */
		virtual INNLayer* CreateLayerGPU(GUID guid)const = 0;
	};

}	// NeuralNetwork
}	// Gravisbell

#endif